import subprocess
import signal
import time

from threading import Timer


currently_logging = True


def run_pcm_commands(
        log_file_path: str,
        logging_interval: int,
        cmd_runtime: int,
        pcm_cmds: list[str]
):
    global currently_logging
    if not currently_logging:
        return

    for pcm_cmd in pcm_cmds:
        log_file_name = log_file_path + f'_{pcm_cmd}'
        print(f'run_pcm_commands pcm_cmd: {pcm_cmd}')
        print(f'run_pcm_commands log_file_name: {log_file_name}')

        with open(log_file_name, 'a') as log_file:
            log_file.write(f'TIMESTAMP: {time.time()}\n')
            full_pcm_cmd = f'sudo {pcm_cmd}'

            try:
                print(f'run_pcm_commands starting pcm_process: {full_pcm_cmd}')
                pcm_process = subprocess.Popen(
                    full_pcm_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f'run_pcm_commands sleeping {cmd_runtime} seconds...')
                time.sleep(cmd_runtime)
                print("Sending SIGINT to pcm_process...")
                pcm_process.send_signal(signal.SIGINT)
                #print(f'run_pcm_commands terminating pcm_process...')
                #pcm_process.terminate()
                #print(f'run_pcm_commands communicating pcm_process...')
                #pcm_process.communicate()

                print(f'run_pcm_commands communicating pcm_process...')
                stdout, stderr = pcm_process.communicate()
                print(f'run_pcm_commands writing stdout...')
                log_file.write(stdout.decode('utf-8') + '\n')
                print('Done.')
            except subprocess.CalledProcessError as e:
                log_file.write(f'Error running {full_pcm_cmd}: {e}\n')

    Timer(
        logging_interval,
        run_pcm_commands,
        args=(
            log_file_path,
            logging_interval,
            cmd_runtime,
            pcm_cmds
        )
    ).start()
