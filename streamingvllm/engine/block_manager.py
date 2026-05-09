from collections import deque
import xxhash
import numpy as np

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash, token_ids):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    @classmethod
    def compute_hash(cls, token_ids, prefix=-1):
        # 🚨 在 Streaming 模式下禁用 Hash 计算，提升性能
        return -1

    def _allocate_block(self, block_id):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq):
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq):
        assert not seq.block_table
        
        # 🚨 彻底关闭 Prefix Cache 匹配逻辑 🚨
        # 在 Streaming 模式下，新分配的序列（如 System Prompt）总是从头开始
        for i in range(seq.num_blocks):
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            
            # 不再计算 Hash，也不更新 hash_to_block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq):
        for bid in reversed(seq.block_table):
            b = self.blocks[bid]
            b.ref_count -= 1
            if b.ref_count == 0:
                self._deallocate_block(bid)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq):
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq):
        bt = seq.block_table
        lb = self.blocks[bt[-1]]
        
        # 🚨 同样关闭 Append 时的 Hash 计算 🚨
        if len(seq) % self.block_size == 1:
            bid = self.free_block_ids[0]
            self._allocate_block(bid)
            bt.append(bid)
        # elif len(seq) % self.block_size == 0 的分支直接省略，因为不需要更新 Hash

    def release_blocks(self, block_ids):
        for bid in block_ids:
            if bid not in self.used_block_ids:
                continue
            b = self.blocks[bid]
            b.ref_count -= 1
            if b.ref_count == 0:
                # 不再需要从 hash_to_block_id 中删除
                self._deallocate_block(bid)
