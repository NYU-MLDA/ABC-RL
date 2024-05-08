import numpy as np


class ReplayMemory:
    """
    Replay memory storing experience tuples from previous episodes. Once can
    obtain batches of experience tuples from the replay memory for training
    a policy network. Old tuples are overwritten once memory limit is
    reached.
    """

    def __init__(self, size, column_types, column_shapes=None, batch_size=32):
        """
        :param size: Number of experience tuples to be stored in the replay
        memory.
        :param column_types: Dictionary mapping column names to data type of
        data to be stored in it.
        :param column_shapes: Shapes of the data stored in the cells of each
        column as lists. Empty list for scalar values. If None, all shapes are
        considered scalar.
        :param batch_size: Size of the batch which is sampled from the replay
        memory.
        """

        self.size = size
        self.batch_size = batch_size
        self.string_column_name = "ob"

        # Convert shape of individual rows to shape of entire columns.
        #if column_shapes is None:
        #    column_shapes = {key: [] for key in column_types.keys()}
        column_shapes = {key: [self.size]+shape for (key, shape)
                         in column_shapes.items()}

        # Preallocate memory
        self.columns = {}
        for column_name, data_type in column_types.items():
            if data_type == "string":
                self.columns[column_name] = ["" for i in range(self.size)]
            else:
                self.columns[column_name] = np.empty(column_shapes[column_name],dtype=data_type)

        # Index of the row in the replay memory which gets replaced during the
        # next insert.
        self.current = 0
        # Total number of rows stored in the replay memory.
        self.count = 0

    def add(self, row):
        """
        Add new row of data to replay memory.
        :param row: Dictionary containing the new row's data for each column.
        """
        for column_name in self.columns.keys():
            self.columns[column_name][self.current] = row[column_name]

        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def add_all(self, rows):
        """
        Add multiple new rows of data to replay memory.
        :param rows: Dictionary of lists containing the new rows' data for each
        column.
        """
        # assert len(actions) == len(rewards) == len(obs) == len(values)
        num = len(list(rows.values())[0])
        assert all(len(x) == num for x in rows.values())

        if self.current + num <= self.size:
            num_free = self.size - self.current
            for column_name in self.columns.keys():
                if column_name != self.string_column_name:
                    #print(column_name)
                    self.columns[column_name][list(np.arange(num)+self.current)] = \
                        rows[column_name]
                else:
                    for i in range(num):
                        self.columns[column_name][self.current+i] = \
                        rows[column_name][i]
        else:
            num_free = self.size - self.current
            num_over = num - num_free
            # Insert first few elements at the end
            for column_name in self.columns.keys():
                if column_name != self.string_column_name:
                    self.columns[column_name][self.current:] = \
                        rows[column_name][:num_free]
                else:
                    for i in range(num_free):
                        self.columns[column_name][self.current+i] = \
                            rows[column_name][i]
            # Insert remaining elements at the front
            for column_name in self.columns.keys():
                if column_name != self.string_column_name:
                    self.columns[column_name][:num_over] = \
                        rows[column_name][num_free:]
                else:
                    for i in range(num_over):
                        self.columns[column_name][i] = \
                            rows[column_name][num_free+i]

        self.count = max(self.count, min(self.current + num, self.size))
        self.current = (self.current + num) % self.size

    def get_minibatch(self,order=0):
        """
        Returns a batch of experience tuples for training.
        :return: Dictionary containing a numpy array for each column.
        """
        if order == 0:
            #indices = np.random.choice(self.count, self.batch_size)
            indices = np.arange(0,self.size)
            #np.append(indices,np.random.choice(self.count, self.batch_size-self.size))
        else:
            starIdx = (order-1)*self.batch_size
            endIdx = starIdx+self.batch_size
            if endIdx < self.count:
                indices = np.arange(starIdx,endIdx)
            else:
                indices = np.random.choice(self.count, self.batch_size)
        minibatch = {}
        for column_name in self.columns.keys():
            if column_name == self.string_column_name:
                minibatch[column_name] = [self.columns[column_name][i] for i in indices]
            else:
                minibatch[column_name] = self.columns[column_name][indices]

        return minibatch

    def __str__(self):
        descr = ""
        for column_name in self.columns.keys():
            descr += "{0}: {1}\n".format(column_name,
                                         self.columns[column_name].__str__())
        return descr


if __name__ == '__main__':
    # mem = ReplayMemory(10, {"state": np.uint8, "value": np.float32},
    #                   column_shapes={"state": [], "value": [2]},
    #                   batch_size=2)
    mem = ReplayMemory(10, {"state": "string", "value": np.float32},
                       column_shapes={"state": [], "value": [2]},
                       batch_size=2)
    #mem.add({"state": 2, "value": [0.5, 0.5]})
    #mem.add({"state": 3, "value": [1.0, 1.0]})
    mem.add({"state": "xyzPath", "value": [1.0, 1.0]})
    mem.add({"state": "abcPath", "value": [1.5, 1.0]})
    print(mem)
    print()
    #mem.add_all({"state": [4, 5], "value": [[1.5, 1.5], [2.0, 2.0]]})
    mem.add_all({"state": ["xyzPath","abcPath"], "value": [[1.5, 1.5], [2.0, 2.0]]})
    print(mem)
    print(mem.get_minibatch())