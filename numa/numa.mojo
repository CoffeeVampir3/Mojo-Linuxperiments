import linux.sys as linux
from std.pathlib import Path
from .cpumask import CpuMask

def get_current_cpu_and_node() -> Tuple[Int, Int]:
    var sys = linux.linux_sys()
    return sys.sys_getcpu()

def read_sysfs(path: String) raises -> String:
    var p = Path(path)
    if not p.exists():
        return String("")
    var content = p.read_text()
    var newline_pos = content.find("\n")
    if newline_pos >= 0:
        return String(content[byte=:newline_pos])
    return content

def parse_cpulist(cpulist: String) raises -> List[Int]:
    var cpus = List[Int]()
    if len(cpulist) == 0:
        return cpus^
    var parts = cpulist.split(",")
    for part in parts:
        var dash_pos = part.find("-")
        if dash_pos >= 0:
            var start = atol(String(part[byte=:dash_pos]))
            var end = atol(String(part[byte=dash_pos + 1:]))
            for cpu in range(start, end + 1):
                cpus.append(cpu)
        else:
            cpus.append(atol(String(part)))
    return cpus^

def parse_distances(s: String) raises -> List[Int]:
    var distances = List[Int]()
    var parts = s.split(" ")
    for part in parts:
        if len(part) > 0:
            distances.append(atol(String(part)))
    return distances^

def parse_meminfo(path: String, field: String) raises -> Int:
    var p = Path(path)
    if not p.exists():
        return 0
    var content = p.read_text()
    var lines = content.split("\n")
    for line in lines:
        if field in line:
            var key_pos = line.find(field)
            if key_pos == -1:
                continue
            var bytes = line.as_bytes()
            var value_start = key_pos + len(field)

            while value_start < len(bytes):
                var b = bytes[value_start]
                if b == Byte(32) or b == Byte(58) or b == Byte(9):
                    value_start += 1
                else:
                    break

            var value = 0
            var saw_digit = False
            var value_end = value_start
            while value_end < len(bytes):
                var b = bytes[value_end]
                if b >= Byte(48) and b <= Byte(57):
                    value = value * 10 + Int(b - Byte(48))
                    saw_digit = True
                    value_end += 1
                else:
                    break
            if saw_digit:
                return value
    return 0

@fieldwise_init
struct NumaNode(Copyable, Writable):
    var id: Int
    var cpu_ids: List[Int]
    var distances: List[Int]
    var mem_total_kb: Int
    var mem_free_kb: Int

    def __init__(out self, id: Int):
        self.id = id
        self.cpu_ids = List[Int]()
        self.distances = List[Int]()
        self.mem_total_kb = 0
        self.mem_free_kb = 0

@fieldwise_init
struct NumaTopology(Movable):
    """Ring-ordered NUMA node placement for tensor parallelism.
    Nodes are selected for minimum communication cost and ordered
    by nearest-neighbor adjacency for ring allreduce."""
    var node_ids: List[Int]
    var tp: Int

    def __len__(self) -> Int:
        return self.tp

    def __getitem__(self, rank: Int) -> Int:
        return self.node_ids[rank]


struct NumaInfo:
    var nodes: List[NumaNode]
    var num_nodes: Int

    def __init__(out self):
        self.num_nodes = 0
        self.nodes = List[NumaNode]()
        try:
            var online_str = read_sysfs("/sys/devices/system/node/online")
            if len(online_str) == 0:
                return
            var node_ids = parse_cpulist(online_str)
            for node_id in node_ids:
                var base = "/sys/devices/system/node/node" + String(node_id)
                var node = NumaNode(node_id)
                node.cpu_ids = parse_cpulist(read_sysfs(base + "/cpulist"))
                node.distances = parse_distances(read_sysfs(base + "/distance"))
                node.mem_total_kb = parse_meminfo(base + "/meminfo", "MemTotal")
                node.mem_free_kb = parse_meminfo(base + "/meminfo", "MemFree")
                self.nodes.append(node^)
                self.num_nodes += 1
        except:
            print("NumaInfo failed to read system numa information or it was not present on the system.")

    def get_node_cpus(self, node_id: Int) -> List[Int]:
        """Get the list of CPU IDs belonging to the specified NUMA node."""
        if node_id < 0 or node_id >= self.num_nodes:
            return List[Int]()
        return self.nodes[node_id].cpu_ids.copy()

    def get_node_mask[mask_size: Int = 128](self, node_id: Int) -> CpuMask[mask_size]:
        var mask = CpuMask[mask_size]()
        if node_id < 0 or node_id >= self.num_nodes:
            return mask^
        for cpu in self.nodes[node_id].cpu_ids:
            mask.set(cpu)
        return mask^

    def distance(self, from_node: Int, to_node: Int) -> Int:
        """Get the NUMA distance between two nodes."""
        if from_node < 0 or from_node >= self.num_nodes:
            return -1
        if to_node < 0 or to_node >= len(self.nodes[from_node].distances):
            return -1
        return self.nodes[from_node].distances[to_node]

    def cpus_per_node(self) -> Int:
        """Get the number of CPUs per node (assumes uniform topology)."""
        if self.num_nodes == 0:
            return 0
        return len(self.nodes[0].cpu_ids)

    def cpus_on_node(self, node: Int) -> Int:
        """Get the number of CPUs on a specific node."""
        if node < 0 or node >= self.num_nodes:
            return 0
        return len(self.nodes[node].cpu_ids)

    def total_cpus(self) -> Int:
        """Get the total number of CPUs across all nodes."""
        var total = 0
        for node in self.nodes:
            total += len(node.cpu_ids)
        return total

    def plan_topology(self, tp: Int) -> NumaTopology:
        """Select tp NUMA nodes with minimum communication cost, ordered as
        a nearest-neighbor ring for optimal allreduce adjacency.

        Two phases:
        1. Greedy selection: seed with the most central node (minimum total
           distance to all others), then greedily add the node closest to
           the selected set. O(tp^2 * num_nodes).
        2. Ring ordering: nearest-neighbor TSP starting from the seed,
           producing the ring traversal order. O(tp^2).

        Returns a NumaTopology with node IDs in ring order — rank 0 is the
        seed (most central of the selected set), and each subsequent rank
        is adjacent in the communication ring.
        """
        if self.num_nodes <= 1 or tp <= 1:
            var ids = List[Int]()
            var node_id = self.nodes[0].id if self.num_nodes > 0 else 0
            for _ in range(tp):
                ids.append(node_id)
            return NumaTopology(ids^, tp)

        # --- Phase 1: Greedy selection from topological center ---

        # Find the most central node (minimum total distance to all others).
        var best_centrality = Int.MAX
        var seed = 0
        for i in range(self.num_nodes):
            var total = 0
            for j in range(self.num_nodes):
                total += self.distance(i, j)
            if total < best_centrality:
                best_centrality = total
                seed = i

        var selected = List[Bool](length=self.num_nodes, fill=False)
        var chosen = List[Int]()
        selected[seed] = True
        chosen.append(seed)

        # Greedily add the node with minimum distance to any already-selected node.
        while len(chosen) < tp and len(chosen) < self.num_nodes:
            var best_node = -1
            var best_dist = Int.MAX
            for candidate in range(self.num_nodes):
                if selected[candidate]:
                    continue
                var min_dist = Int.MAX
                for s in range(len(chosen)):
                    var d = self.distance(candidate, chosen[s])
                    if d < min_dist:
                        min_dist = d
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_node = candidate
            if best_node < 0:
                break
            selected[best_node] = True
            chosen.append(best_node)

        # --- Phase 2: Nearest-neighbor ring ordering ---

        var ordered = List[Int]()
        var visited = List[Bool](length=len(chosen), fill=False)

        # Start from the seed (index 0 in chosen).
        visited[0] = True
        ordered.append(self.nodes[chosen[0]].id)

        for step in range(1, len(chosen)):
            var last = chosen[0]
            # Find which chosen[] index corresponds to the last ordered node.
            for k in range(len(chosen)):
                if self.nodes[chosen[k]].id == ordered[step - 1]:
                    last = chosen[k]
                    break

            var best_next = -1
            var best_d = Int.MAX
            for k in range(len(chosen)):
                if visited[k]:
                    continue
                var d = self.distance(last, chosen[k])
                if d < best_d:
                    best_d = d
                    best_next = k
            if best_next >= 0:
                visited[best_next] = True
                ordered.append(self.nodes[chosen[best_next]].id)

        return NumaTopology(ordered^, tp)

    def print_debug(self):
        print("NUMA Info:", self.num_nodes, "nodes,", self.cpus_per_node(), "cpus/node")
        print()
        for i in range(self.num_nodes):
            print("Node", i, ":", len(self.nodes[i].cpu_ids), "cpus,", self.nodes[i].mem_total_kb // 1024, "MB total,", self.nodes[i].mem_free_kb // 1024, "MB free")
        print()
        print("Distance matrix:")
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                print(self.distance(i, j), end=" ")
            print()
