class MobileDevice:
    def __init__(self,
                 md_id,
                 local_compute_power,
                 uplink_rate):

        self.id = md_id

        # MFLOPs/s
        self.local_compute_power = local_compute_power

        # KB/s
        self.uplink_rate = uplink_rate

        self.associated_server = None