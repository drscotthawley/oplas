from vicregaddon import vicreg_var_loss, vicreg_inv_loss, vicreg_cov_loss
import torch 
import torch.nn.functional as F
from .mixing import mix_and_encode, do_encode



def vicreg_loss_fn(x, x2, gamma=1.0, eps=1e-5, 
                lambda_v=1e-5, lambda_i=1.0, lambda_c=0.1,  # regularization/tuning parameters
                extra_info=True, # return sub-losses as dict 
                ):
    """VICReg loss function.  Combines the three VICReg loss functions into one"""
    var_loss = vicreg_var_loss(x, gamma, eps) * lambda_v
    inv_loss = vicreg_inv_loss(x, x2) * lambda_i
    cov_loss = vicreg_cov_loss(x) * lambda_c
    return {'var_loss':var_loss, 'inv_loss':inv_loss, 'cov_loss':cov_loss}



def mag_loss_fn(z,
                lambda_m:float=.01,   # scale factor
                ):
    """intended to 'gently' keep vector norms in the vicinity of 1 without 'forcing' it"""
    phi = z.norm(dim=-1)
    #return lambda_m * (phi**4 - 2*phi**2 + 1 ).mean()  # phi^4 or 'mexican hat potential'
    #return lambda_m * ((phi + 1)*(phi -1)**2).mean() # alternate; not an even function but doens't shoot high as quick for phi>1, and gradient is nonzero near phi=0
    return lambda_m * (phi - 1).pow(2).mean()  # simpler version, just a parabola around phi=1


def get_loss_on_batch(batch, 
                    epoch,
                    projector, 
                    encoder, 
                    start_inverse_epoch=10, 
                    device='cuda',                     
                    ):
    """Your full loss-calulation routine
    'audio' can come from train or val dataset.
    """ 
    audio = batch.to(device)

    with torch.no_grad(): # encoding is frozen  
        ret = mix_and_encode(audio, encoder)
        ys, y_mix = ret['ys'], ret['y_mix']
        y_sum = 0
        for y in ys:
            y_sum = y_sum + y   # just for checking linearity later
        #y_sum /= 2  # average of the embeddings just for the heck of it

    # project via h
    zs, y_hats = [], []
    for y in ys:
        z, y_hat = projector(y)
        zs.append(z)
        y_hats.append(y_hat)

    z_mix, y_mix_hat = projector(y_mix)

    z_sum = 0 
    for z in zs: 
        z_sum += z 
    #z_sum /= 2 # average of the embeddings just for the heck of it

    losses  = vicreg_loss_fn(z_sum, z_mix)            # sum of embeddings = embedding of mix 

    #disable inversion loss and/or magnitude loss until other parts are known to work
    if True:
        # if epoch >= start_inverse_epoch:  # wait a bit before even bothering to try to learn an inverse
        #     inverse_loss = 0
        #     for y_hat,y in zip(y_hats, ys):
        #         inverse_loss = inverse_loss + F.mse_loss(y_hat, y)  # don't need vicreg on y_hatb/c inverse is "supervised learning"
        #     inverse_loss += F.mse_loss(y_mix_hat, y_mix)  
        #     losses = losses | {'inverse_loss':inverse_loss}

        #mag_loss = torch.mean([sombrero_loss_fn(y_hat) for y_hat in y_hats+[y_mix]])  # CLAP embeddings should be unit vectors, but the inverse_loss should handle that anyway
        mag_loss = torch.stack([mag_loss_fn(z) for z in [z_mix,z_sum]]).mean() # keep z's, zsum, mix  mag's kinda near 1
        losses = losses | {'mag_loss':mag_loss}

    info = {'zs':zs, 'z_sum':z_sum, 'z_mix':z_mix, 'ys':ys, 'y_mix':y_mix, 'y_hats':y_hats, 'y_mix_hat':y_mix_hat, 'y_sum':y_sum }
    return losses, info
