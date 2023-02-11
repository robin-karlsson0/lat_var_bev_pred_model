def integrate_obs_and_lat_pred(x_obs,
                               x_pred,
                               m_obs,
                               threshold=0.499,
                               is_np=False,
                               override_pred=False):
    '''
    Integrate binary semantic information from observation and prediction.

    Args:
        x_obs: Real observation tensor (B, C, H, W).
        x_pred: Prediction tensor (B, C, H, W).
        m_obs: Boolean mask indicating observed elements (B, C, H, W).
        threshold: Elements above will be presumed 'True' for semantic.
    '''
    if is_np:
        x_int = x_pred.copy()
    else:
        x_int = x_pred.clone()

    if override_pred:
        x_int[m_obs] = x_obs[m_obs]

    mask = x_int < threshold
    x_int[mask] = 0

    mask = x_int > threshold
    x_int[mask] = 1

    return x_int
