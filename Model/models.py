import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                if dropout: fcs.append(torch.nn.Dropout(p=0.2))
                if batchnorm: fcs.append(torch.nn.BatchNorm1d(sizes[i]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x): 
        return self.layers(x) 
    

class TimingGNN(torch.nn.Module):
    def __init__(self, bd_nf, bw_nf, fw_nf, out_bd, out_bw, out_fw):
        super(TimingGNN, self).__init__()
        self.bd_nf = bd_nf  # bidirectional node feature dimension
        self.bw_nf = bw_nf # backward node feature dimension
        self.fw_nf = fw_nf # forward node feature dimension
        self.out_bd = out_bd
        self.out_bw = out_bw
        self.out_fw = out_fw

        # MLP for forward and backward propogation and self awaration
        self.MLP_msg_forward = MLP(self.bd_nf + self.fw_nf, 64, 64, 1 + self.out_bd + self.out_fw)
        self.MLP_msg_backward = MLP(self.bd_nf + self.bw_nf, 64, 64, 1 + self.out_bd + self.out_bw)
        self.MLP_msg_self = MLP(self.bd_nf + self.fw_nf + self.bw_nf, 64, 64, self.out_fw + self.out_bd + self.out_bw)


    # forward propagation
    def message_func_forward(self, edges):
        src_features = torch.cat([edges.src['fw_nf'], edges.src['bd_nf']], dim=1)
        x = self.MLP_msg_forward(src_features)
        k, f1, f2 = torch.split(x, [1, self.out_bd, self.out_fw], dim=1)
        k = torch.sigmoid(k)
        return {'msg_f1': f1 * k, 'msg_f2': f2 * k}
    
    # backward propagation
    def message_func_backward(self, edges):
        dst_features = torch.cat([edges.src['bw_nf'], edges.src['bd_nf']], dim=1)
        x = self.MLP_msg_backward(dst_features)
        k, b1, b2 = torch.split(x, [1, self.out_bd, self.out_bw], dim=1)
        k = torch.sigmoid(k)
        return {'msg_b1': b1 * k, 'msg_b2': b2 * k}

    # Node feature aggregation (for cellarc and netarc)
    def node_reduce_fw(self, nodes):
        out_bd1 = torch.mean(nodes.mailbox['msg_f1'], dim=1)
        out_fw = torch.mean(nodes.mailbox['msg_f2'], dim=1)
        return {'out_bd1': out_bd1, 'out_fw': out_fw}
    
    def node_reduce_bw(self, nodes):
        out_bd2 = torch.mean(nodes.mailbox['msg_b1'], dim=1)
        out_bw = torch.mean(nodes.mailbox['msg_b2'], dim=1)
        return {'out_bd2': out_bd2, 'out_bw': out_bw}

    def forward(self, g, bd_nf, bw_nf, fw_nf):
        with g.local_scope():
            g.ndata['fw_nf'] = fw_nf
            g.ndata['bw_nf'] = bw_nf
            g.ndata['bd_nf'] = bd_nf

            # forward propagation
            g.update_all(self.message_func_forward, self.node_reduce_fw)
            # backward propagation
            g.update_all(self.message_func_backward, self.node_reduce_bw)

            node_self_feature = torch.cat([g.ndata['fw_nf'], g.ndata['bw_nf'], g.ndata['bd_nf']], dim=1)
            node_self_feature = self.MLP_msg_self(node_self_feature)
            out_bd, out_bw, out_fw = torch.split(node_self_feature, [self.out_bd, self.out_bw, self.out_fw], dim=1)

            g.ndata['out_bd'] = torch.sigmoid(out_bd + g.ndata['out_bd1'] + g.ndata['out_bd2'])
            g.ndata['out_bw'] = torch.sigmoid(out_bw + g.ndata['out_bw'])
            g.ndata['out_fw'] = torch.sigmoid(out_fw + g.ndata['out_fw'])
            
            return g.ndata['out_bd'], g.ndata['out_bw'], g.ndata['out_fw']


class MultiLayerTimingGNN(torch.nn.Module):
    def __init__(self, num_layers, out_nf):
        super(MultiLayerTimingGNN, self).__init__()
        self.num_layers = num_layers
        self.gnn_layers = torch.nn.ModuleList()
        self.out_nf = out_nf

        # First layer
        self.gnn_layers.append(TimingGNN(4, 2, 1, 40, 16, 8))

        # Middle layers
        for _ in range(1, num_layers):
            self.gnn_layers.append(TimingGNN(40, 16, 8, 40, 16, 8))

        # Last layer
        self.FC_reduce = torch.nn.Linear(64, self.out_nf)

    def forward(self, g):
        with g.local_scope():
            fw_nf = g.ndata['forward_feature']
            bw_nf = g.ndata['backward_feature']
            bd_nf = g.ndata['bidirection_feature']

            # Propagate through remaining layers
            for i in range(0, self.num_layers):
                bd_nf, bw_nf, fw_nf = self.gnn_layers[i](g, bd_nf, bw_nf, fw_nf)
            
            nf = torch.cat([bd_nf, bw_nf, fw_nf], dim=1)
            nf = self.FC_reduce(nf)

            return nf



class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # input 4*64*64
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)  # output 32*64*64
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # output 64*32*32
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)  # output 32*16*16
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)   # output 1*16*16
        
        self.fc = torch.nn.Linear(1 * 16 * 16, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # output 1*16*16
        
        x = x.view(x.size(0), -1)  # output (batch_size, 1*16*16)
        
        x = self.fc(x)  # output (batch_size, 64)
        return x
    

# class MultiModalNNWithGate(torch.nn.Module):
#     def __init__(self, num_layers, hidden_nf, out_nf, output, embedding_dim, num_heads=4, num_Attention_layer=3, in_nf = 5, in_ef = 4, h1=32, h2=32, in_channels=4):
#         super(MultiModalNN, self).__init__()
#         self.gnn = MultiLayerTimingGNN(num_layers, hidden_nf, out_nf, in_nf, in_ef, h1, h2)
#         self.cnn = CNN(in_channels)
#         self.AttentionPool = SelfAttentionPool(embedding_dim, num_heads, num_Attention_layer)
#         self.MLP_cnn_forward = MLP(64*64, 64, 64, 32)   # MLP after Padding
#         self.MLP_gnn_forward = MLP(out_nf, 64, 64, 32) # MLP after pooling
#         self.MLP_gsize_forward = MLP(embedding_dim, 64, 64, 32) # MLP after SelfAttentionPool
#         self.MLP_CPath_Gate = MLP(15, 64, 64, 32) # MLP for Gate feature on CPath
#         self.MLP_Output = MLP(128, 64, 64, output)
        
#     def forward(self, g, img, padding_mask, Gate_feature, Gate_size):
#         # Ensure the data is in batch form
#         if not hasattr(g, 'batch_size'):
#             g = dgl.batch([g])
#         if img.dim() == 3:
#             img = img.unsqueeze(0)
#         if padding_mask.dim() == 3:
#             padding_mask = padding_mask.unsqueeze(0)
#         if Gate_feature.dim() == 1:
#             Gate_feature = Gate_feature.unsqueeze(0)
#         if Gate_size.dim() == 2:
#             Gate_size = Gate_size.unsqueeze(0)
        
#         # GNN part: process the batched graph
#         with g.local_scope():
#             nf = self.gnn(g)
#             padding_mask_expanded = g.ndata['padding_mask'].unsqueeze(-1)
#             g.ndata['feature'] = nf * padding_mask_expanded
#             sum_nf = dgl.sum_nodes(g, 'feature')
#             num_masked_nodes = dgl.sum_nodes(g, 'padding_mask').unsqueeze(-1)
#             pooled_nf = sum_nf / (num_masked_nodes + 1e-6) # avoide division by zero

#         # CNN part: process the batch of images
#         img_batch_size = img.size(0)  # Assume img is of shape (batch_size, channels, height, width)
#         img = self.cnn(img)  # Process each image
#         img = torch.mul(img, padding_mask) 
#         img = img.view(img_batch_size, -1)
        
#         # Gate size part: process the gate size sequence
#         gsize = self.AttentionPool(Gate_size)
#         gsize = self.MLP_gsize_forward(gsize)
        
#         Img_feature = self.MLP_cnn_forward(img)
#         Gnn_feature = self.MLP_gnn_forward(pooled_nf)
#         Gate_feature = self.MLP_CPath_Gate(Gate_feature)

#         # Concatenate features across the batch
#         combined_features = torch.cat([Img_feature, Gnn_feature, gsize, Gate_feature], dim=1)  # Concatenate along feature dim
#         Output = self.MLP_Output(combined_features)
#         return Output
    
# class MultiModalNN(torch.nn.Module): # without gate feature
#     def __init__(self, num_layers, hidden_nf, out_nf, output, embedding_dim, num_heads=4, num_Attention_layer=3, in_nf = 5, in_ef = 4, h1=32, h2=32, in_channels=4):
#         super(MultiModalNN, self).__init__()
#         self.gnn = MultiLayerTimingGNN(num_layers, hidden_nf, out_nf, in_nf, in_ef, h1, h2)
#         self.cnn = CNN(in_channels)
#         self.AttentionPool = SelfAttentionPool(embedding_dim, num_heads, num_Attention_layer)
#         self.MLP_cnn_forward = MLP(64*64, 64, 64, 32)   # MLP after Padding
#         self.MLP_gnn_forward = MLP(out_nf, 64, 64, 32) # MLP after pooling
#         self.MLP_gsize_forward = MLP(embedding_dim, 64, 64, 32) # MLP after SelfAttentionPool
#         self.MLP_Output = MLP(96, 64, 64, output)
        
#     def forward(self, g, img, padding_mask, Gate_size):
#         # Ensure the data is in batch form
#         if not hasattr(g, 'batch_size'):
#             g = dgl.batch([g])
#         if img.dim() == 3:
#             img = img.unsqueeze(0)
#         if padding_mask.dim() == 3:
#             padding_mask = padding_mask.unsqueeze(0)
#         if Gate_size.dim() == 2:
#             Gate_size = Gate_size.unsqueeze(0)
        
#         # GNN part: process the batched graph
#         with g.local_scope():
#             nf = self.gnn(g)
#             padding_mask_expanded = g.ndata['padding_mask'].unsqueeze(-1)
#             g.ndata['feature'] = nf * padding_mask_expanded
#             sum_nf = dgl.sum_nodes(g, 'feature')
#             num_masked_nodes = dgl.sum_nodes(g, 'padding_mask').unsqueeze(-1)
#             pooled_nf = sum_nf / (num_masked_nodes + 1e-6) # avoide division by zero

#         # CNN part: process the batch of images
#         img_batch_size = img.size(0)  # Assume img is of shape (batch_size, channels, height, width)
#         img = self.cnn(img)  # Process each image
#         img = torch.mul(img, padding_mask) 
#         img = img.view(img_batch_size, -1)
        
#         # Gate size part: process the gate size sequence
#         gsize = self.AttentionPool(Gate_size)
#         gsize = self.MLP_gsize_forward(gsize)
        
#         Img_feature = self.MLP_cnn_forward(img)
#         Gnn_feature = self.MLP_gnn_forward(pooled_nf)

#         # Concatenate features across the batch
#         combined_features = torch.cat([Img_feature, Gnn_feature, gsize], dim=1)  # Concatenate along feature dim
#         Output = self.MLP_Output(combined_features)
#         return Output