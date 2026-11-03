"""Execute a UTF-8 shell script on the authorized GPU server using the existing SSH key."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('script', type=Path)
args = parser.parse_args()
script = args.script.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
command = ['ssh', '-i', str(Path.home() / '.ssh/object_detection_cloud'), '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-p', '13549', 'root@connect.nmb2.seetacloud.com', 'bash', '-s']
raise SystemExit(subprocess.run(command, input=script.encode('utf-8')).returncode)
