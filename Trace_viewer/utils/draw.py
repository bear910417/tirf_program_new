import plotly.graph_objects as go


def draw(fig, tot_bkps, i, time_g, color):
    for bkp in range(len(tot_bkps[i])):
        try:
            fig.add_shape( type='rect', x0=time_g[int(tot_bkps[i][bkp])], x1=time_g[int(tot_bkps[i][bkp+1])], y0=0,y1=1,fillcolor=color[bkp%2],opacity=0.1,line_width=0,yref='y2 domain')
            fig.add_shape( type='rect', x0=time_g[int(tot_bkps[i][bkp])], x1=time_g[int(tot_bkps[i][bkp+1])], y0=0,y1=1,fillcolor=color[bkp%2],opacity=0.1,line_width=0,yref='y3 domain') 
            fig.add_shape( type='rect', x0=time_g[int(tot_bkps[i][bkp])], x1=time_g[int(tot_bkps[i][bkp+1])], y0=0,y1=1,fillcolor=color[bkp%2],opacity=0.1,line_width=0,yref='y domain') 
       
        except:
            fig.add_shape( type='rect',x0=time_g[int(tot_bkps[i][bkp])], x1=time_g[-1] ,y0=0, y1=1,fillcolor=color[bkp%2],opacity=0.1,line_width=0,yref='y2 domain')  
            fig.add_shape( type='rect',x0=time_g[int(tot_bkps[i][bkp])], x1=time_g[-1] ,y0=0, y1=1,fillcolor=color[bkp%2],opacity=0.1,line_width=0,yref='y3 domain')
            fig.add_shape( type='rect',x0=time_g[int(tot_bkps[i][bkp])], x1=time_g[-1] ,y0=0, y1=1,fillcolor=color[bkp%2],opacity=0.1,line_width=0,yref='y domain')
            
    return fig
