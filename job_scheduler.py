import schedule
import subprocess
import time

def job():
    subprocess.run(['heat_mcmc', '--rdir', '/opt/data', '-f', '-p', '-d'])
    subprocess.run(['heat_uks', '--rdir', '/opt/data',
                    '-f', '-p', '-d',
                    '--benchmark', '/opt/data/mcmc_sampling.nc'])

schedule.every().day.at("11:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(10)
