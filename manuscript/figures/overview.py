import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import StamenTerrain
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

fn = '6277_RCL_Karen_Britten_GNS_Science.jpg'
arr_img = plt.imread(fn, format='jpg')
y, x, z = arr_img.shape
factor=300
fig = plt.figure(figsize=(x/factor, y/factor))
ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.9])
ax1.imshow(arr_img)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.text(1850,750, 'O', color='red', weight='bold', size='18')

# Map
tiler = StamenTerrain()
mercator = tiler.crs
ax2 = fig.add_axes([0.64,0.1,0.3,0.35], projection=mercator)
ax2.set_extent([165, 180, -49, -32], crs=ccrs.PlateCarree())
ax2.add_image(tiler, 6)
ax2.coastlines(resolution='50m')

label_style = {'color': 'black', 'weight': 'bold', 'size':10,
               'bbox': {'facecolor':'white', 'edgecolor': 'white', 'visible': True}}
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                     linewidth=1, color='gray', alpha=0.5,
                                     linestyle='--', xlabel_style=label_style,
                                     ylabel_style=label_style)
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlines = True
gl.ylines = True
gl.xlocator = mticker.FixedLocator([160, 166, 172, 179, 180])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
ax2.plot(175.564490, -39.281149, marker='^', color='red',
                  markersize=6, transform=ccrs.PlateCarree())
fig.savefig('overview.png', dpi=300, bbox_inches='tight')
