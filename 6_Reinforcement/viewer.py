import matplotlib.pyplot as plt
from matplotlib import animation
from JSAnimation.IPython_display import display_animation
from IPython.display import display
import numpy as np
from PIL import Image, ImageDraw


class Viewer(object):
    """
    Custom viewer for OpenAI Gym Env.

    env.render() uses pyglet, which requires GLX > 1.2, which is cumbersome
    to get working on a server.
    """

    def __init__(self, env, custom_render=False):
        self.env = env.unwrapped
        self.custom_render = custom_render
        self.frames = []

    def render(self, close=False, display_gif=False):
        """Render and store frame."""
        # if 'close' try to close window
        if close:
            try:
                self.env.render(close=True)
            except:
                pass
        # else render frame
        else:
            if not self.custom_render:
                try:
                    # try env.render
                    img = self.env.render(mode='rgb_array')
                except:
                    # if error, switch to custom render
                    self.custom_render = True
            if self.custom_render:
                if str(type(self.env).__name__) == 'CartPoleEnv':
                    img = self._render_cartpole(np.array(self.env.state))
                else:
                    raise Exception('Unable to render environment!')
            self.frames.append(img)
        if display_gif:
            self._display_gif()

    def _render_cartpole(self, state):
        """Custom render for cartpole environment."""
        screen_width = 600
        screen_height = 400
        x = int(state[0] * 50 + screen_width/2.0) # horisontal position
        y = int(screen_height * 0.7) # vertical position
        theta = state[2] # angles in radians away from horizontal
        cart_h = int(screen_height * 0.1)
        cart_w = int(screen_width * 0.08)
        pole_h = int(screen_height * 0.45)
        pole_w = int(screen_width * 0.02)
        image = Image.fromarray(np.zeros((screen_height, screen_width)) + 250)
        draw = ImageDraw.Draw(image)
        draw.line((0,y,screen_width,y), width=2, fill=0) # horisontal line
        draw.line((x-cart_w, y, x+cart_w, y), width=cart_h, fill=128) # cart
        draw.line((x - np.sin(theta) * pole_w * 0.5, # x coordinate of bottom of pole
                   y + np.cos(theta) * pole_w * 0.5, # y coordinate of bottom of pole
                   x + np.sin(theta) * pole_h, # x coordinate of top of pole
                   y - np.cos(theta) * pole_h), # y coordinate of top of pole
                   width=pole_w, fill=0)
        draw.ellipse((x - pole_w * 0.2,
                      y - pole_w * 0.2,
                      x + pole_w * 0.2,
                      y + pole_w * 0.2),
                      fill=128)
        del draw
        return image

    def _display_gif(self):
        """Displays list of frames as a gif, with controls."""
        patch = plt.imshow(self.frames[0])
        plt.axis('off')
        anim = animation.FuncAnimation(plt.gcf(), lambda i: patch.set_data(self.frames[i]), frames=len(self.frames), interval=20)
        display(display_animation(anim, default_mode='once'))
