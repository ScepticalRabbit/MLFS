# -*- coding: utf-8 -*-
"""
TODO
"""

class PlotProps:
    def __init__(self):
        self.aspect_ratio = 1.62     
        self.a4_width = 8.25
        self.a4_height = 11.75
        self.a4_margin_width = 1
        self.a4_margin_height = 1
        self.a4_print_width = self.a4_width-2*self.a4_margin_width 
        self.a4_print_height = self.a4_height-2*self.a4_margin_height
        
        self.single_fig_scale = 0.65
        self.single_fig_size = (self.a4_print_width*self.single_fig_scale,
                                self.a4_print_width*self.single_fig_scale/self.aspect_ratio)
        
        self.resolution = 300
        
        self.font_name = 'Liberation Sans'
        self.font_def_weight = 'normal'
        self.font_def_size = 10
        self.font_tick_size = 9
        self.font_head_size = 12
        self.font_ax_size = 11
        self.font_leg_size = 9
        
        self.ms = 5.0
        self.lw = 1.0
        
        self.cmap_seq = "cividis"
        self.cmap_div = "RdBu"
