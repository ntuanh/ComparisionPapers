class Compute:
    def __init__(self):
        pass

    # FLOPs
    def Fl(self , din, dout ):
        """
        FLOPs for fully connected
        :param din: dimension input
        :param dout: dimension output
        :return:
        """
        return 2*din*dout   # (din + (din - 1) + 1) * dout

    def Fc(self , cin , cout , kw, kh, w, h):
        """
        FLOPs for convolution kernels
        :param cin: channel size
        :param cout:
        :param kw: kernel size
        :param kh:
        :param w: output size
        :param h:
        :return: int
        """
        return 2*cin*kw*kh*cout*w*h

    def Dl(self , cin, w, h):
        """
        Compute output data size D_L (in MB)
        Assumes FP32 (4 bytes per element)
        """
        bytes_per_element = 4  # FP32
        return cin * w * h * bytes_per_element / (1024 * 1024)



