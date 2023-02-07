import schedule
import subprocess
import time

def job():
    subprocess.run(['heat_uks', '--rdir', '/opt/data', '-s', '2016-03-04', '-f', '-p', '-d'])

schedule.every().day.at("11:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(10)
