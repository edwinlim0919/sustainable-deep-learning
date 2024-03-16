import subprocess
import time

from threading import Timer


def log_pcm(
        log_file_path: str,
        interval: int,
        pcm_cmds: list[str]
):
    timestamp = time.time()

    for pcm_cmd in pcm_cmds:
        log_file_name = log_file_path + f'_{pcm_cmd}'

        with open(log_file_name, 'a') as log_file:
            log_file.write(f'TIMESTAMP: {timestamp}\n')
            full_pcm_cmd = f'sudo {pcm_cmd}'

            try:
                result = subprocess.run(
                    full_pcm_cmd,
                    shell=True,
                    check=True,
                    text=True,
                    capture_output=True
                )
                log_file.write(result.stdout + '\n')
            except subprocess.CalledProcessError as e:
                log_file.write(f'Error running {full_pcm_cmd}: {e}\n')

    Timer(
        interval,
        run_pcm_commands,
        args=(log_file_path, interval, pcm_cmds)
    )
