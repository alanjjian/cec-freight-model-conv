import collections

"""
Author: Alan Jian
This file contains the ComputeNode object class specification.
"""


class NodeDict(collections.defaultdict):

    def __getattr__(self, attribute_name):
        """Enables us to use dot notation to access an item in this
        NodeDict, and returns an Attribute Error if its not
        present. Always maps a string to a Node."""
        if attribute_name in self.keys():
            return self[attribute_name]
        raise AttributeError(attribute_name)

    def __setattr__(self, key, value):
        raise AttributeError(
            (
                "Attributes of a NodeDict cannot be modified. Use built-in"
                "ComputeGraph methods to add, modify, and delete Nodes "
                "instead."
            )
        )


class ComputeNode:
    """
    These nodes wrap our functions, allowing us to avoid redundancies in our
    computation, and not worry about the overall structure of our ComputeGraph
    when adding complexity.

    They allow us to visualize our model as a computation graph (much like
    Analytica), and run computationally intensive steps in parallel to
    maximize efficiency (this functionality is provided by dask).

    The compute node pressumes a particular workflow:
    1. Identify a node that needs additional work.
    2. Rework the function within that node, creating additional dependencies.
    3. For each additional dependency, either identify an existing node that
       satisfies it, or build a new ComputeNode.

    Additional Restrictions:
    1. Each ComputeNode can only provide one output (i.e a dataframe, an array,
       etc.). Multi-output ComputeNodes are not supported.
    """

    input_args: list[str]  # should be overridden

    def __init__(self):
        """
        Initialize a ComputeNode for use in a Computation Graph. For a
        functional compute node, you'll need:

        function: this describes the computation that will be executed at this
        node.
        inputs: a dictionary describing the argname (key) and the
        corresponding ComputeNode (value) that will
        be responsible for calculating the dependency.

        Internals:

        self._input_mapping: dictionary describing the argname (key) and the
        corresponding ComputeNode (value) that will be responsible for
        calculating the dependency.
        self._output: the value of the computation executed by this node.
        """

        self._input_mapping = NodeDict(int)
        self._rev_input_mapping = NodeDict(int)
        self._output_nodes = []
        self._output = None

    def function(self):
        raise NotImplementedError(
            "implement computation function when using ComputeNode."
        )

    # INPUTS ###

    def get_input_args(self):
        """Returns a list of input arguments as specified by function."""
        return self.input_args

    def get_input_nodes(self):
        """Returns a list of mapped input nodes."""
        return self._input_mapping.values()

    def get_input_node(self, arg: str):
        """Returns the ComputeNode that corresponds to ARG."""
        if not self.is_mapped_arg(arg):
            raise AttributeError("No such argument")
        return self._input_mapping[arg]

    def get_input_arg(self, node):
        """Returns the arg that corresponds to NODE."""
        if not self.is_mapped_node(node):
            raise AttributeError("This compute node isn't mapped.")
        return self._rev_input_mapping[node]

    def get_input_mapping(self):
        """Returns the dictionary that maps our each arg to a ComputeNode."""
        return self._input_mapping

    def is_arg(self, arg: str) -> bool:
        """Returns whether ARG is an argument of the function wrapped by this
        ComputeNode."""
        return arg in self.get_input_args()

    def is_mapped_arg(self, arg: str) -> bool:
        """Returns boolean of whether an ARG is mapped."""
        return isinstance(self._input_mapping[arg], ComputeNode)

    def is_mapped_node(self, node) -> bool:
        """Returns boolean of whether an ARG is mapped."""
        return isinstance(self._rev_input_mapping[node], str)

    def is_completely_mapped(self):
        """Returns bool of whether the inputs required by function are mapped
        for this ComputeNode."""
        return all(
            [
                self.is_arg(arg) and self.is_mapped_arg(arg)
                for arg in self.get_input_args()
            ]
        )

    def add_input_map(self, compute_node, arg):
        """Alias for adding a node edge between two nodes."""
        ComputeNode.add_node_edge(compute_node, self, arg)

    def _add_input_map(self, compute_node, arg):
        """Maps an ARG to its corresponding COMPUTE_NODE. Overwrites existing
        mapping if OVERWRITE is set to true."""
        if self.is_arg(arg):
            self._input_mapping[arg] = compute_node
            self._rev_input_mapping[compute_node] = arg

    def add_input_mappings(self, input_mapping: dict, overwrite=False):
        """Sets input mapping that maps input arg to its corresponding
        ComputeNode as specified by INPUT_MAPPING."""
        for mapping in input_mapping.keys():
            self.add_input_map(mapping, input_mapping[mapping], overwrite)

    def remove_input_map(self, compute_node, arg):
        """Alias for removing a node edge between two nodes."""
        ComputeNode.remove_node_edge(compute_node, self, arg)

    def _remove_input_map(self, compute_node=None, arg=None):
        """Unmaps ARG and its corresponding COMPUTE_NODE."""
        if arg is None and compute_node is None:
            raise LookupError("What are we removing?")
        if arg is None:
            arg = self.get_input_arg(compute_node)
        else:
            compute_node = self.get_input_node(arg)
        if self.is_mapped_arg(arg):
            self._input_mapping.pop(arg)
            self._rev_input_mapping.pop(compute_node)

    def remove_input_mappings(self, input_mapping):
        """Removes input mappings that maps input arg to its corresponding
        ComputeNode as specified by INPUT_MAPPING."""
        for mapping in input_mapping.keys():
            self.remove_input_map(mapping, input_mapping[mapping])

    # OUTPUTS ###

    def _add_output_node(self, output_node):
        """Adds an output ComputeNode that will receive information from this
        ComputeNode."""
        if not self.has_output_node(output_node):
            self._output_nodes.append(output_node)

    def _remove_output_node(self, output_node):
        """Removes OUTPUT_NODE from this node."""
        if self.has_output_node(output_node):
            self._output_nodes.remove(output_node)

    def get_output_nodes(self):
        """Returns the nodes that are using the outputs of this ComputeNode."""
        return self._output_nodes

    def has_output_node(self, output_node):
        """Returns whether this node's outputs already have OUTPUT_NODE."""
        return output_node in self._output_nodes

    # Edge Manipulation

    @staticmethod
    def add_node_edge(
        output_node,
        input_node,
        input_arg: str,
    ):
        """Updates OUTPUT_NODE's outputs to include INPUT_NODE, and updates
        INPUT_NODE'S inputs to include OUTPUT_NODE mapped to INPUT_ARG,
        overwriting existing mappings if OVERWRITE is true."""
        output_node._add_output_node(input_node)
        input_node._add_input_map(output_node, input_arg)

    @staticmethod
    def remove_node_edge(
        output_node=None,
        input_node=None,
        input_arg=None,
    ):
        """Updates OUTPUT_NODE's outputs by removing INPUT_NODE, and updates
        INPUT_NODE'S inputs to exclude OUTPUT_NODE mapped to INPUT_ARG,
        overwriting existing mappings if OVERWRITE is true."""
        if not output_node:
            if input_arg:
                output_node = input_node.get_input_node(input_arg)
            else:
                raise KeyError(
                    "No input argument or output node specified to remove edge"
                )
        output_node._remove_output_node(input_node)
        input_node._remove_input_map(compute_node=output_node)

    @staticmethod
    def disconnect_node(compute_node):
        """Removes all of COMPUTE_NODE edges, and cleans up adjacent
        nodes."""
        # Follow outputs to unmap their inputs
        output_nodes = compute_node.get_output_nodes().copy()
        for output_node in output_nodes:
            ComputeNode.remove_node_edge(compute_node, output_node)
        # Follow inputs to unmap their outputs
        input_args = list(compute_node.get_input_mapping().keys())
        for input_arg in input_args:
            ComputeNode.remove_node_edge(
                input_node=compute_node, input_arg=input_arg
            )

    def disconnect(self):
        ComputeNode.disconnect_node(self)

    # Computation ###

    @staticmethod
    def get_upstream_compute_order(compute_node):
        """Returns list of nodes upstream of and including COMPUTE_NODE sorted
        in topologic order."""

        # Navigate to the top of DAG, add nodes with in-degree 0 to a queue.
        visited = collections.defaultdict(int)
        indegree_dict = {}
        q = collections.deque()  # queue for initial traversal
        topo_q = collections.deque()  # queue for topo sort
        q.appendleft(compute_node)
        while True:
            if len(q) == 0:
                break
            curr_node = q.pop()
            if visited[curr_node]:
                continue
            visited[curr_node] = 1
            input_nodes = curr_node.get_input_nodes()
            indegree_dict[curr_node] = len(input_nodes)
            if indegree_dict[curr_node] == 0:
                topo_q.appendleft(curr_node)
            else:
                q.extendleft(input_nodes)

        # Topologic Sort
        # Note: `visited` also lists all of COMPUTE_NODE's dependencies.
        topo_lst = []
        while True:
            if len(topo_q) == 0:
                break
            curr_node = topo_q.pop()
            topo_lst.append(curr_node)
            output_nodes = curr_node.get_output_nodes()
            for output_node in output_nodes:
                if visited[output_node]:  # checks if its a dependency
                    indegree_dict[output_node] -= 1
                    if indegree_dict[output_node] == 0:
                        topo_q.appendleft(output_node)

        if sum(indegree_dict.values()) > 0:
            raise Exception("Cycle is present in graph")

        return topo_lst

    def get_compute_order(self):
        return ComputeNode.get_upstream_compute_order(self)

    def compute(self, compute_node):
        """Returns result of running COMPUTE_NODE and its upstream nodes in
        topologic order. Intermediate results are stored in each node."""

        # Get topologic sorted order of nodes
        topo_lst = self.get_compute_order(compute_node)

        # Execute each function in order
        for compute_node in topo_lst:
            compute_node.compute()
