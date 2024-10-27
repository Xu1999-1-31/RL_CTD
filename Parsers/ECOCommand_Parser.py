class Size_Command:
    def __init__(self, name, command):
        self.name = name
        self.command = []
        self.command.append(command)
    def __repr__(self):
        return (f'Cell name: {self.name}, command: {self.command}')

def Read_ECOCommand(infile):
    sized_cell_command = {}
    with open(infile, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            index = line.split()
            if len(index) != 0:
                if index[0] == 'size_cell':
                    try:
                        new_size
                    except NameError:
                        pass
                    else:
                        sized_cell_command[new_size.name] = new_size.command
                    new_size = Size_Command(index[1].replace('{', '').replace('}', ''), line)
                elif index[0] == 'set_cell_location':
                    new_size.command.append(line)
    return sized_cell_command