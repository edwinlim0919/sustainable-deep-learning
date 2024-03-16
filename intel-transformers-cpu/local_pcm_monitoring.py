import subprocess
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

        with open(log_file_name, 'a') as log_file:
            log_file.write(f'TIMESTAMP: {time.time()}\n')
            full_pcm_cmd = f'sudo {pcm_cmd}'

            try:
                pcm_process = subprocess.Popen(
                    full_pcm_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(cmd_runtime)
                pcm_process.terminate()

                stdout, stderr = pcm_process.communicate()
                log_file.write(stdout.decode('utf-8') + '\n')
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
    )
