classdef ReplayBuffer < handle
    properties
        capacity
        buffer
        ptr
    end
    
    methods
        function self = ReplayBuffer(capacity)
            self.capacity = capacity;
            self.buffer = cell(capacity, 1);
            self.ptr = 1;
        end
        
        function add(self, experience)
            self.buffer{self.ptr} = experience;
            self.ptr = mod(self.ptr, self.capacity) + 1;
        end
        
        function batch = sample(self, batch_size)
            current_size = sum(~cellfun(@isempty, self.buffer));
            indices = randi(current_size, batch_size, 1);
            batch = self.buffer(indices);
        end
        
        function ready = is_ready(self, batch_size)
            ready = sum(~cellfun(@isempty, self.buffer)) > batch_size;
        end
    end
end