from datetime import date, timedelta
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import xarray as xr

from pumahu.uks import main as main_uks

DATADIR = '/opt/data/'
DATAFILE = os.path.join(DATADIR, 'uks.nc')
if not os.path.isfile(DATAFILE):
    main_uks(['--rdir', DATADIR, '-s', '2016-03-04', '-f'])

app = FastAPI()
# -- allow any origin to query API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
)

host = "volcanolab.gns.cri.nz"
port = "11111"

@app.get("/")
async def root():
    examplelinkdate = f"http://{host}:{port}/v1/data/?start=2022-01-01&end=2023-01-01"
    html_content = f"""
    <html>
        <head>
            <title>Ruapehu Heat Flow API</title>
        </head>
        <body>
            <h3>Welcome to the GNS Heat Flow API</h3>
            <p>To request heat flow data you have to provide a start and end date that you are interested in.</p>
            <h3>Example API syntax: </h3>
            <p>Requesting heat flow data between 1 January 2022 and 1 January 2023: <a href="{examplelinkdate}" target=_blank>{examplelinkdate}</a></p>
            <p>To load the data into a pandas dataframe do the following: </p>
            <pre class="brush: python">
            import pandas as pd
            df = pd.read_csv(http://{examplelinkdate}, parse_dates=True, index_col=0)
            </pre>

        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/v1/data/")
def root(param: str='q_in',
         start: date=date.today() - timedelta(days=365),
         end: date=date.today()):
    xds = xr.open_dataset(DATAFILE)   
    df = xds.exp.loc[start:end, param].to_pandas()
    df['dates'] = df.index
    output = df.to_csv(index=False, columns=['dates', 'val', 'std'])
    return StreamingResponse(
        iter([output]),
        media_type='text/csv',
        headers={"Content-Disposition":
                 "attachment;filename=<heatflow_RCL>.csv"})

def main(argv=None):
    uvicorn.run(app, host="0.0.0.0", port=8061)

if __name__ == "__main__":
    main()