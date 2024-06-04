"""
routines for visualization / debugging


 point_cloud (tokens, method='pca', color_scheme='batch',
              output_type='wandbobj', mode='markers', size=3,
              line={'color': 'rgba(10,10,10,0.01)'}, ds_preproj=1,
              ds_preplot=1, debug=False, colormap=None, darkmode=False,
              layout_dict=None, **kwargs)
"""
import torch 
from einops import rearrange
import pandas as pd 
import plotly.express as px


#from aeiou.viz import point_cloud, project_down 

# copied from aeiou.viz
def project_down(tokens,     # batched high-dimensional data with dims (b,d,n)
            proj_dims=3,     # dimensions to project to
            method='pca',    # projection method: 'pca'|'umap'
            n_neighbors=10, # umap parameter for number of neighbors
            min_dist=0.3,    # umap param for minimum distance
            debug=False,     # print more info while running
            **kwargs,        # other params to pass to umap, cf. https://umap-learn.readthedocs.io/en/latest/parameters.html 
            ):
    "this projects to lower dimenions, grabbing the first _`proj_dims`_ dimensions"
    method = method.lower()
    if debug: print("tokens.shape =",tokens.shape)
    A = tokens
    #A = tokens.view(-1, tokens.shape[0]) # put batch dimension last
    #A = rearrange(tokens, 'b d n -> (b n) d') # put all the vectors into the same d-dim space
    if A.shape[-1] > proj_dims: 
        if method=='umap':
            proj_data = umap.UMAP(n_components=proj_dims, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric='correlation', **kwargs).fit_transform(A.cpu().numpy())
            proj_data = torch.from_numpy(proj_data).to(tokens.device)
        else:  # pca
            (U, S, V) = torch.pca_lowrank(A)
            proj_data = torch.matmul(A, V[:, :proj_dims])  # this is the actual PCA projection step
    else:
        proj_data = A
    if debug: print("proj_data.shape =",proj_data.shape)
    #return torch.reshape(proj_data, (tokens.size()[0], -1, proj_dims)) # put it in shape [batch, n, proj_dims]
    return proj_data

@torch.no_grad()
def update_df(data, df=None, label='data', debug=False):
    # create a df with columns x, y, z, label.
    if debug: print("data.shape =",data.shape)
    new_df = pd.DataFrame(data.cpu().numpy(), columns=['x', 'y', 'z'])
    new_df['label'] = pd.Series([label] * data.shape[0]) 
    df = new_df if df is None else pd.concat([df, new_df], ignore_index=True)
    if debug: print("df = \n",df)
    return df


@torch.no_grad()
def scatter3d(x_sum, x_mix, xs=None, method='pca', prefix='z', debug=False):
    labels = [f'{prefix}_sum',f'{prefix}_mix']
    data_nd = [x_sum, x_mix]
    if xs is not None:
        labels = labels + [f'{prefix}_{i}' for i in range(len(xs))]
        data_nd = data_nd + xs
    data_nd = torch.stack(data_nd, dim=0).cpu()
    data_nd = data_nd.view(-1, data_nd.shape[-1])
    if debug: print("\ndata_nd.shape =",data_nd.shape)
    data_3d = project_down(data_nd, method=method) 
    # project all data to 3D (or 2D but cmon) & add to dataframe
    df = None 
    stride = len(x_sum)
    for i in range(len(labels)): 
        dat_3d = data_3d[i*stride:(i+1)*stride]
        if debug: print("dat_3d.shape =",dat_3d.shape)
        df = update_df(dat_3d, df=df, label=labels[i]) 
    if debug: print("scatter3d: df = \n",df)
    # make the graph
    fig = px.scatter_3d(df, x='x', y='y', z='z',color='label')
    fig.update_layout( margin=dict(l=20, r=20, t=20, b=20),)
    fig.update_traces(marker_size=3)
    return fig 




@torch.no_grad()
def viz_sums_mixes(x_sum, x_mix, xs=None, method='pca', prefix='z', debug=False, sum_fac=1.0):
    if debug: print("\nviz_sums_mixes: x_sum.shape = ",x_sum.shape)
    if sum_fac != 1.0: 
        x_sum = x_sum * sum_fac 
    graph = scatter3d(x_sum, x_mix, xs=xs, method=method, prefix=prefix)
    # if xs is None:
    #     data = torch.stack([x_sum.T/3.95, x_mix.T, ], dim=0)
    # else: 
    #     data = torch.stack([x_sum.T/4, x_mix.T] + [x.T for x in xs], dim=0)
    # if debug: print("viz_sums_mixes: data.shape = ",data.shape)
    # graph = point_cloud(data, method=method, color_scheme='batch', output_type='plotly', mode='markers', size=3, debug=False,)

    return graph 



