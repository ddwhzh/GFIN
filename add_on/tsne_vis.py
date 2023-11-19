from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np 

from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.preprocessing import StandardScaler

def tsne_vis(featureList ,labelsList, num_cls):
    #   归一化
    featureList_scaled = featureList.cpu()
    # scaler=StandardScaler()
    # featureList_scaled = scaler.fit_transform(featureList_scaled)

    x_encode = TSNE(n_components=2).fit_transform(featureList_scaled) # 接着使用tSNE进行降维
    # x_encode = scaler.fit_transform(x_encode)
    #print(x_encode.shape)
    # 进行可视化
    cmap = plt.get_cmap('plasma', num_cls) # 数字与颜色的转换
    # 获得可视化数据
    v_x = x_encode
    v_y = labelsList.cpu()
    # 进行可视化
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    classes = range(num_cls)
    for key in classes:
        ix = np.where(v_y==key)
        ax.scatter(v_x[ix][:,0], v_x[ix][:,1], color=cmap(key), label=key)
        # ax.text(np.mean(v_x[ix][:,0]), np.mean(v_x[ix][:,1]), key, 
        #         fontsize=18, bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()
    plt.show()

    canvas = FigureCanvasAgg(fig)
    # Do some plotting here
    # ax = fig.gca()
    # ax.plot([1, 2, 3])
    # ax.set_axis_off()
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    fig_np = np.asarray(buf)
    fig_np = np.expand_dims(fig_np[:,:, :3].transpose(2,0,1), 0)
    return fig, fig_np