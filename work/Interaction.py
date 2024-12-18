import subprocess
import shutil
import signal
import os
import atexit
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var
import ReBuildPtScripts

def check_command(command_name):
    # check if the command is available
    return shutil.which(command_name) is not None

def Run_Pt_Script(script):
    """
    Run a PrimeTime (pt_shell) script using subprocess, ensuring child processes
    are terminated when the main process exits or is killed.

    Args:
        script (str): The name of the script to run.
    """
    if not check_command('pt_shell'):
        raise EnvironmentError("Error: 'pt_shell' not found. Please ensure PrimeTime is correctly installed and added to your PATH.")
    
    command = ['pt_shell', '-f', '../' + script]
    working_directory = os.path.join(Global_var.work_dir, 'log/')
    
    # Create the working directory if it doesn't exist
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    
    print(f'Running PT script: {script} in {working_directory}')
    
    # Start the subprocess and assign it to a new process group
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Register cleanup to terminate child processes on exit
    def cleanup():
        if process.poll() is None:  # Check if the process is still running
            print("Terminating PT subprocess...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Terminate the process group
            print("PT subprocess terminated.")
    
    atexit.register(cleanup)  # Ensure cleanup is called on normal program exit

    # Handle signals such as Ctrl+C
    def handle_signal(signum, frame):
        print(f"Received signal {signal.Signals(signum).name}, cleaning up...")
        cleanup()
        exit(0)

    signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Handle kill signals

    try:
        # Read and print the subprocess output and error streams
        for line in process.stdout:
            print(line, end='')  # Print standard output
        for err_line in process.stderr:
            print(err_line, end='')  # Print standard error
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating PT script...")
        cleanup()
    finally:
        process.wait()  # Wait for the process to complete
    
    # Check the subprocess return code
    print('Return code:', process.returncode)


def Run_Icc2_Script(script):
    """
    Run an IC Compiler II (icc2_shell) script using subprocess, ensuring child processes
    are terminated when the main process exits or is killed.

    Args:
        script (str): The name of the script to run.
    """
    if not check_command('icc2_shell'):
        raise EnvironmentError("Error: 'icc2_shell' not found. Please ensure IC Compiler II is correctly installed and added to your PATH.")
    
    command = ['icc2_shell', '-f', '../' + script]
    working_directory = os.path.join(Global_var.work_dir, 'log/')
    
    # Create the working directory if it doesn't exist
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    
    print(f'Running ICC2 script: {script} in {working_directory}')
    
    # Start the subprocess and assign it to a new process group
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Register cleanup to terminate child processes on exit
    def cleanup():
        if process.poll() is None:  # Check if the process is still running
            print("Terminating ICC2 subprocess...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Terminate the process group
            print("ICC2 subprocess terminated.")
    
    atexit.register(cleanup)  # Ensure cleanup is called on normal program exit

    # Handle signals such as Ctrl+C
    def handle_signal(signum, frame):
        print(f"Received signal {signal.Signals(signum).name}, cleaning up...")
        cleanup()
        exit(0)

    signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Handle kill signals

    try:
        # Read and print the subprocess output and error streams
        for line in process.stdout:
            print(line, end='')  # Print standard output
        for err_line in process.stderr:
            print(err_line, end='')  # Print standard error
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating ICC2 script...")
        cleanup()
    finally:
        process.wait()  # Wait for the process to complete
    
    # Check the subprocess return code
    print('Return code:', process.returncode)


def Write_Pt_Cells_Scripts(design, verbose=False):
    file = os.path.join(Global_var.work_dir, 'pt_rpt.tcl')
    with open(file, 'r') as file:
        script_content = file.read()
    updated_content = script_content.replace('set top_design aes_cipher_top', f'set top_design {design}')
    updated_content = script_content.replace('update_timing -full', '#update_timing -full')
    updated_content = updated_content.replace('report_timing -nosplit -nets -input_pins -transition_time -capacitance -significant_digit 6 -max_path 100000 > ../PtRpt/${top_design}.rpt',
                                            '#report_timing -nosplit -nets -input_pins -transition_time -capacitance -significant_digit 6 -max_path 100000 > ../PtRpt/${top_design}.rpt')
    updated_content = updated_content.replace('report_global_timing -significant_digits 8 > ../PtRpt/${top_design}_global.rpt',
                                            '#report_global_timing -significant_digits 8 > ../PtRpt/${top_design}_global.rpt')
    updated_content = updated_content.replace('if {[file exists ../Delay_scripts/${top_design}_Delay.tcl]} {',
                                            '#if {[file exists ../Delay_scripts/${top_design}_Delay.tcl]} {')
    updated_content = updated_content.replace('    source ../Delay_scripts/${top_design}_Delay.tcl > ../PtRpt/${top_design}_Delay.rpt',
                                            '#    source ../Delay_scripts/${top_design}_Delay.tcl > ../PtRpt/${top_design}_Delay.rpt')
    updated_content = updated_content.replace('}\n', '#}\n')
    script_filename = os.path.join(Global_var.work_dir, f'{design}_pt_rpt.tcl')
    with open(script_filename, 'w') as new_file:
        new_file.write(updated_content)

    if verbose:
        print(f"Generated PT Cells script for {design}")
    
def Write_Pt_Scripts(design, ECO=True, VerilogInline=False, verbose=False):
    if ECO and VerilogInline:
        raise ValueError("ECO and VerilogInline cannot both be True at the same time.")
    # read initial file
    if VerilogInline:
        file = os.path.join(Global_var.work_dir, 'pt_rpt_inline.tcl')
    else:
        file = os.path.join(Global_var.work_dir, 'pt_rpt.tcl')
    with open(file, 'r') as file:
        script_content = file.read()

    # replace initial design to new design
    if ECO:
        updated_content = script_content.replace('set top_design aes_cipher_top', f'set top_design {design}_eco')
        updated_content = updated_content.replace('../Delay_scripts/${top_design}_Delay.tcl', f'../Delay_scripts/{design}_Delay.tcl')
    else:
        updated_content = script_content.replace('set top_design aes_cipher_top', f'set top_design {design}')

    # write new file
    script_filename = os.path.join(Global_var.work_dir, f'{design}_pt_rpt.tcl')
    with open(script_filename, 'w') as new_file:
        new_file.write(updated_content)
    
    if verbose:
        print(f"Generated PT script for {design}")
        
def Write_Icc2_Scripts(design, verbose=False):
    # read initial file
    file = os.path.join(Global_var.work_dir, 'icc2_rpt.tcl')
    with open(file, 'r') as file:
        script_content = file.read()

    # replace initial design to new design
    updated_content = script_content.replace('set bench aes_cipher_top', f'set bench {design}')

    # write new file
    script_filename = os.path.join(Global_var.work_dir, f'{design}_icc2_rpt.tcl')
    with open(script_filename, 'w') as new_file:
        new_file.write(updated_content)
    
    if verbose:
        print(f"Generated ICC2 script for {design}")

def Write_Icc2_ECO_Scripts(design, verbose=False):
    # read initial file
    file = os.path.join(Global_var.work_dir, 'icc2_eco.tcl')
    with open(file, 'r') as file:
        script_content = file.read()

    # replace initial design to new design
    updated_content = script_content.replace('set bench aes_cipher_top', f'set bench {design}')

    # write new file
    script_filename = os.path.join(Global_var.work_dir, f'{design}_icc2_eco.tcl')
    with open(script_filename, 'w') as new_file:
        new_file.write(updated_content)
    
    if verbose:
        print(f"Generated ICC2 ECO script for {design}")

def Delete_Temp_Scripts(design, verbose=False):
    path = os.path.join(Global_var.work_dir, f'{design}_icc2_rpt.tcl')
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(Global_var.work_dir, f'{design}_icc2_eco.tcl')
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(Global_var.work_dir, f'{design}_pt_rpt.tcl')
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(Global_var.work_dir, f'{design}_pt_incremental_eco.tcl')
    if os.path.exists(path):
        os.remove(path)
    if verbose:
        print(f"Deleted temporary scripts for {design}")
        
def ECO_PRPT_Iteration(design, verbose=False): # one eco iteriation: incremental PR and PT
    Write_Icc2_ECO_Scripts(design, verbose)
    Run_Icc2_Script(f'{design}_icc2_eco.tcl')
    Write_Pt_Scripts(design, True, False, verbose) # Write PT script for ECO
    Run_Pt_Script(f'{design}_pt_rpt.tcl')
    Delete_Temp_Scripts(design, verbose)

def VerilogInline_PT_Iteration(design, verbose=False):
    Write_Pt_Scripts(design, False, True, verbose) # Write PT script for Verilog Inline change
    Run_Pt_Script(f'{design}_pt_rpt.tcl')
    Delete_Temp_Scripts(design, verbose)

def VerilogInlineChange(design, cells, Incremental=False, verbose=False): # cells:[name: [initial cell, final cell]]
    # read initial file
    if Incremental:
        file = os.path.join(Global_var.work_dir, 'VerilogInline/' + design + '_route.v')
    else:
        file = os.path.join(Global_var.work_dir, 'Icc2Output/' + design + '_route.v')
    with open(file, 'r') as infile:
        content = infile.read()
    # save the original verilog
    file = os.path.join(Global_var.work_dir, 'VerilogInline/' + design +'_route.v.bak')
    with open(file, 'w') as outfile:
        outfile.write(content)
    for key, value in cells.items():
        content = content.replace(value[0]+' '+key+' (', value[1]+' '+key+' (')
    file = os.path.join(Global_var.work_dir, 'VerilogInline/' + design + '_route.v')
    with open(file, 'w') as outfile:
        outfile.write(content)
    if verbose:
        print(f"Changed Verilog Inline for {design}")

def VerilogInlineBackspace(design):
    src_file = os.path.join(Global_var.work_dir, 'VerilogInline/' + design + '_route.v.bak')
    dst_file = os.path.join(Global_var.work_dir, 'VerilogInline/' + design + '_route.v')
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)

def  Write_Incremental_ECO_Scripts(design, cellLists, verbose=False): # write changelist for icc2; cellLists: [{cellName: [initial cell, final cell], ...}, ...]
    path = os.path.join(Global_var.work_dir, f'ECO_ChangeList/{design}_dt_eco.tcl')
    with open(path, 'w') as file:
        file.write(f'current_instance\n')
        for celldict in cellLists:
            for key, value in celldict.items():
                file.write(f'size_cell {{{key}}} {{{value[1]}}}\n')
    if verbose:
        print(f"Generated Incremental ECO script for {design}")
        
def Write_Pt_Incremental_ECO_Scripts(design):
    script_filename = os.path.join(Global_var.work_dir, f'{design}_pt_incremental_eco.tcl')
    with open(script_filename, 'w') as outfile:
        outfile.write('restore_session ../Saved_session/' + design + '/\n')
        outfile.write('source ../ECO_ChangeList/' + design + '_dt_eco.tcl\n')
        outfile.write('report_timing -nosplit -nets -input_pins -transition_time -capacitance -significant_digit 6 -max_path 100000 > ../PtRpt/' + design + '_inline.rpt\n')
        outfile.write('report_global_timing -significant_digits 8 > ../PtRpt/' + design + '_inline_global.rpt\n')
        outfile.write('report_cell -connections -nosplit > ../PtRpt/' + design + '_inline_cell.rpt\n')
        outfile.write('if {[file exists ../Delay_scripts/' + design + '_Delay.tcl]} {\n')
        outfile.write('    source ../Delay_scripts/' + design + '_Delay.tcl > ../PtRpt/' + design + '_inline_Delay.rpt\n}\n')
        outfile.write('exit')

def PT_Incremental_ECO_Iteration(design, verbose=False):
    Write_Pt_Incremental_ECO_Scripts(design)
    Run_Pt_Script(f'{design}_pt_incremental_eco.tcl')
    Delete_Temp_Scripts(design, verbose)

def Write_Icc2_ECO_Command(design, commands):
    path = os.path.join(Global_var.work_dir, f'ECO_ChangeList/{design}_ctd_eco.tcl')
    with open(path, 'w') as file:
        file.write(f'current_instance\n')
        for command in commands:
            file.write(command)

def RLCTD_Iteration(design, verbose=False): # one eco iteriation: incremental PR and PT
    Write_Icc2_ECO_Scripts(design, verbose)
    Run_Icc2_Script(f'{design}_icc2_eco.tcl')
    Write_Pt_Scripts(design, True, False, verbose) # Write PT script for ECO
    Run_Pt_Script(f'{design}_pt_rpt.tcl')
    Delete_Temp_Scripts(design, verbose)
    
def ReBuildAllRpt(design, verbose=False):
    Write_Icc2_Scripts(design, verbose)
    Run_Icc2_Script(f'{design}_icc2_rpt.tcl')
    Write_Pt_Cells_Scripts(design, verbose)
    Run_Pt_Script(f'{design}_pt_rpt.tcl')
    ReBuildPtScripts.ReBuildPtScripts(design)
    Write_Pt_Scripts(design, ECO=False, VerilogInline=False, verbose=verbose)
    Run_Pt_Script(f'{design}_pt_rpt.tcl')
    Delete_Temp_Scripts(design, verbose)

if __name__ == "__main__": 
    Write_Icc2_Scripts('mc_top')
    Run_Icc2_Script('mc_top_icc2_rpt.tcl')
    Write_Pt_Scripts('mc_top', ECO=False, VerilogInline=False)
    Run_Pt_Script('mc_top_pt_rpt.tcl')
    Delete_Temp_Scripts('mc_top')