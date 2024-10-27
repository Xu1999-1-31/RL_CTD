class Sized_Cell:
    def __init__(self, name, sizedtype):
        self.name = name
        self.sizedtype = sizedtype
    def __repr__(self):
        return (f'Cell name: {self.name}, sizedtype: {self.sizedtype}')

def Read_ECOChangeList(infile):
    sized_cell_list = []
    with open(infile, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            index = line.split()
            if len(index) != 0:
                if index[0] == 'size_cell':
                    sized_cell = Sized_Cell(index[1].replace('{', '').replace('}', ''), index[2].replace('{', '').replace('}', ''))
                    sized_cell_list.append(sized_cell)
    
    return sized_cell_list