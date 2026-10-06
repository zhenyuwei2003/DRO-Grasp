import torch


SUPPORTED_DEVICE_TYPES = ("auto", "xpu", "cuda", "cpu")


def _is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def resolve_device(device_type="auto", device_index=0):
    device_type = str(device_type).lower()
    if device_type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"Unsupported device type {device_type!r}; expected one of {SUPPORTED_DEVICE_TYPES}"
        )

    if device_type == "auto":
        if _is_xpu_available():
            device_type = "xpu"
        elif torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

    if device_type == "xpu" and not _is_xpu_available():
        raise RuntimeError(
            "Intel XPU was requested but is unavailable. Install the PyTorch XPU build "
            "and verify that the Intel GPU driver is available."
        )
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, int(device_index))


def empty_device_cache(device):
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize_device(device):
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def get_device_name(device):
    if device.type == "xpu":
        return torch.xpu.get_device_name(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def lightning_device_config(device_type, device_indices):
    device = resolve_device(device_type, device_indices[0] if device_indices else 0)
    if device.type == "cpu":
        return "cpu", 1
    if device.type == "xpu":
        if len(device_indices) != 1:
            raise ValueError("Lightning training currently supports exactly one XPU device")
        from lightning_fabric.accelerators import Accelerator

        class XPUAccelerator(Accelerator):
            def setup_device(self, selected_device):
                if selected_device.type != "xpu":
                    raise ValueError(f"Device should be XPU, got {selected_device} instead")
                torch.xpu.set_device(selected_device)

            def teardown(self):
                torch.xpu.empty_cache()

            @staticmethod
            def parse_devices(devices):
                if isinstance(devices, int):
                    devices = [devices]
                if not isinstance(devices, (list, tuple)) or len(devices) != 1:
                    raise ValueError("XPU training requires one device index")
                return [int(devices[0])]

            @staticmethod
            def get_parallel_devices(devices):
                return [torch.device("xpu", index) for index in devices]

            @staticmethod
            def auto_device_count():
                return torch.xpu.device_count()

            @staticmethod
            def is_available():
                return _is_xpu_available()

        return XPUAccelerator(), [device.index]
    return device.type, list(device_indices)