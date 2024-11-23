import csv
import torch
import math

class filewriter:
    counter = 0
    def __init__(self,layer_count):
        self.buffer = []
        self.handle = open('out/hitmap-%03d.csv' % (layer_count), "w")
        self.writer = csv.writer(self.handle)

    def write(self, data):
        self.buffer.append(data)
        if len(self.buffer) > 100:
            self.flush()

    def flush(self):
        self.writer.writerows(self.buffer)
        self.handle.flush()
        self.buffer.clear()

    def __del__(self):
        self.flush()
        self.handle.close()


class cache_impl:
    layer_count = 0
    def __init__(self, quant_factor=0.80, max_cache_size=150):
        self.cache = {}
        self.random_matrix = None
        self.init_cache = False
        self.hit_count = 0
        self.miss_count = 0
        self.quant_factor = quant_factor
        self.max_cache_size = max_cache_size
        cache_impl.layer_count += 1
        self.count = cache_impl.layer_count

    def forward(self, input):
        if not self.init_cache:
            self.writer = filewriter(self.count)
            ishape = input.shape[-1]
            self.random_matrix = torch.rand(
                (ishape, math.ceil(ishape * self.quant_factor)), device=input.device
            ) - 0.5
            self.init_cache = True
            self.cache_tensor = torch.empty(
                0, math.ceil(ishape * self.quant_factor),
                dtype=torch.int8, device=input.device
            )
            
        
        # Calculate a hash for the input tensor
        signature = input @ self.random_matrix
        signature = signature.flatten(start_dim=0, end_dim=2)
        signature = signature < 0

        if self.cache_tensor.numel() > 0:
            is_cached = torch.any(
                (self.cache_tensor == signature.unsqueeze(1)).all(dim=2), dim=1
            )
        else:
            is_cached = torch.zeros(
                signature.shape[0], dtype=torch.bool, device=signature.device
            )

        self.hit_count += is_cached.sum().item()
        self.miss_count += (~is_cached).sum().item()
        # print(self.hit_count,self.miss_count, self.hit_count/(self.hit_count+self.miss_count))
        # Add misses to the cache
        if (~is_cached).any():
            new_entries = signature[~is_cached]

        # Enforce cache size limit
            if new_entries.size(0) > self.max_cache_size:
                new_entries = new_entries[:self.max_cache_size]

            if self.cache_tensor.size(0) + new_entries.size(0) > self.max_cache_size:
                overflow = (self.cache_tensor.size(0) + new_entries.size(0)) - self.max_cache_size
                self.cache_tensor = self.cache_tensor[overflow:]  # Remove oldest entries

            self.cache_tensor = torch.cat((self.cache_tensor, new_entries), dim=0)

        self.writer.write(
            list(input.shape)
            + list(signature.shape)
            + [self.hit_count, self.miss_count]
            + is_cached.int().tolist()
        )