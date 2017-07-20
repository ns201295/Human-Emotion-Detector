#!/usr/bin/env python


from abc import ABCMeta, abstractmethod
import time

import wx
import cv2


class BaseLayout(wx.Frame):

    __metaclass__ = ABCMeta

    def __init__(self, capture, title=None, parent=None, id=-1, fps=10):

        self.capture = capture
        self.fps = fps

        # determine window size and init wx.Frame
        success, frame = self._acquire_frame()
        if not success:
            print "Could not acquire frame from camera."
            raise SystemExit

        self.imgHeight, self.imgWidth = frame.shape[:2]
        self.bmp = wx.BitmapFromBuffer(self.imgWidth, self.imgHeight, frame)
        wx.Frame.__init__(self, parent, id, title,
                          size=(self.imgWidth, self.imgHeight))

        self._init_base_layout()
        self._create_base_layout()

    def _init_base_layout(self):

        # set up periodic screen capture
        self.timer = wx.Timer(self)
        self.timer.Start(1000./self.fps)
        self.Bind(wx.EVT_TIMER, self._on_next_frame)

        # allow for custom modifications
        self._init_custom_layout()


    def _create_base_layout(self):

        # set up video stream
        self.pnl = wx.Panel(self, size=(self.imgWidth, self.imgHeight))
        self.pnl.SetBackgroundColour(wx.BLACK)
        self.pnl.Bind(wx.EVT_PAINT, self._on_paint)

        # display the button layout beneath the video stream
        self.panels_vertical = wx.BoxSizer(wx.VERTICAL)
        self.panels_vertical.Add(self.pnl, 1, flag=wx.EXPAND | wx.TOP,
                                 border=1)

        # allow for custom layout modifications
        self._create_custom_layout()

        # round off the layout by expanding and centering
        self.SetMinSize((self.imgWidth, self.imgHeight))
        self.SetSizer(self.panels_vertical)
        self.Centre()


    def _on_next_frame(self, event):
        success, frame = self._acquire_frame()
        if success:
            # process current frame
            frame = self._process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # update buffer and paint (EVT_PAINT triggered by Refresh)
            self.bmp.CopyFromBuffer(frame)
            self.Refresh(eraseBackground=False)

    def _on_paint(self, event):
        # read and draw buffered bitmap
        deviceContext = wx.BufferedPaintDC(self.pnl)
        deviceContext.DrawBitmap(self.bmp, 0, 0)

    def _acquire_frame(self):
        return self.capture.read()

    @abstractmethod
    def _init_custom_layout(self):
        pass

    @abstractmethod
    def _create_custom_layout(self):
        pass

    @abstractmethod
    def _process_frame(self, frame_rgb):
        pass
