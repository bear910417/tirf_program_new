

def select_good_bad(changed_id, event, i, select_list_g):
    if ('set_good' in changed_id) or (('n_events' in changed_id) and event['key'] == 'z'):
        select_list_g[i] = 0 if select_list_g[i] == 1 else 1
    if ('set_bad' in changed_id) or (('n_events' in changed_id) and event['key'] == 'x'):
        select_list_g[i] = 0 if select_list_g[i] == -1 else -1
    return select_list_g

def select_colocalized(changed_id, event, i, colocalized_list):
    if ('set_colocalized' in changed_id) or (('n_events' in changed_id) and event['key'] == 'c'):
        colocalized_list[i] = 0 if colocalized_list[i] == 1 else 1
    return colocalized_list

def render_good_bad(i, select_list_g):
    white_button_style = {'background-color': '#f0f0f0', 'color': 'black'}
    red_button_style = {'background-color': 'red', 'color': 'white'}
    green_button_style = {'background-color': 'green', 'color': 'white'}
    if select_list_g.shape[0] == 0:
        return white_button_style, white_button_style
    if select_list_g[i] == 1:
        return green_button_style, white_button_style
    elif select_list_g[i] == 0:
        return white_button_style, white_button_style
    else:
        return white_button_style, red_button_style

def render_colocalized(i, colocalized_list):
    white_button_style = {'background-color': '#f0f0f0', 'color': 'black'}
    blue_button_style = {'background-color': 'blue', 'color': 'white'}
    if colocalized_list.shape[0] == 0:
        return white_button_style
    return blue_button_style if colocalized_list[i] == 1 else white_button_style
