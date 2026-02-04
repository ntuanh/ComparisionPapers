class Compute:
    """
    Computation and communication models
    """

    # -----------------------------
    # FLOPs models
    # -----------------------------

    def Fl(self, din, dout):
        """
        FLOPs for fully connected layer
        """
        return 2 * din * dout

    def Fc(self, cin, cout, kw, kh, w, h):
        """
        FLOPs for convolution layer
        """
        return 2 * cin * kw * kh * cout * w * h

    # -----------------------------
    # Output data size (Eq. 3)
    # -----------------------------

    def Dl(self, cin, w, h):
        """
        Output data size in MB (FP32)
        """
        return cin * w * h * 4 / (1024 * 1024)

    # -----------------------------
    # Time models
    # -----------------------------

    def local_time(self, flops, f_local):
        """
        Local execution time
        """
        return flops / f_local

    def edge_time(self, flops, f_edge):
        """
        Edge execution time
        """
        return flops / f_edge

    def uplink_time(self, data_mb, bandwidth):
        """
        Uplink transmission time
        """
        return data_mb / bandwidth
