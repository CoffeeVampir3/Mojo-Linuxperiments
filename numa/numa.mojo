from sys.intrinsics import inlined_assembly
from memory import UnsafePointer, MutUnsafePointer
from pathlib import Path
from .cpumask import CpuMask

fn syscall_getcpu(cpu: MutUnsafePointer[UInt32, MutOrigin.external], node: MutUnsafePointer[UInt32, MutOrigin.external]) -> Int64:
    return inlined_assembly[
        """
        mov $$309, %rax
        xor %rdx, %rdx
        syscall
        """,
        Int64,
        MutUnsafePointer[UInt32, MutOrigin.external], MutUnsafePointer[UInt32, MutOrigin.external],
        constraints = "={rax},{rdi},{rsi},~{rdx},~{rcx},~{r11},~{memory}",
    ](cpu, node)

fn get_current_cpu_and_node() -> Tuple[Int, Int]:
    var cpu = UInt32(0)
    var node = UInt32(0)
    var cpu_ptr = MutUnsafePointer[UInt32, MutOrigin.external](unsafe_from_address=Int(UnsafePointer(to=cpu)))
    var node_ptr = MutUnsafePointer[UInt32, MutOrigin.external](unsafe_from_address=Int(UnsafePointer(to=node)))
    _ = syscall_getcpu(cpu_ptr, node_ptr)
    return Tuple[Int, Int](Int(cpu), Int(node))

def read_sysfs(path: String) -> String:
    var p = Path(path)
    if not p.exists():
        return String("")
    var content = p.read_text()
    var newline_pos = content.find("\n")
    if newline_pos >= 0:
        return String(content[:newline_pos])
    return content

def parse_cpulist(cpulist: String) -> List[Int]:
    var cpus = List[Int]()
    if len(cpulist) == 0:
        return cpus^
    var parts = cpulist.split(",")
    for i in range(len(parts)):
        var part = String(parts[i])
        var dash_pos = part.find("-")
        if dash_pos >= 0:
            var start = atol(String(part[:dash_pos]))
            var end = atol(String(part[dash_pos + 1:]))
            for cpu in range(start, end + 1):
                cpus.append(cpu)
        else:
            cpus.append(atol(part))
    return cpus^

def parse_distances(s: String) -> List[Int]:
    var distances = List[Int]()
    var parts = s.split(" ")
    for i in range(len(parts)):
        var part = String(parts[i])
        if len(part) > 0:
            distances.append(atol(part))
    return distances^

def parse_meminfo(path: String, field: String) -> Int:
    var p = Path(path)
    if not p.exists():
        return 0
    var content = p.read_text()
    var lines = content.split("\n")
    for i in range(len(lines)):
        var line = String(lines[i])
        if field in line:
            var key_pos = line.find(field)
            if key_pos == -1:
                continue
            var value_start = key_pos + len(field)
            while value_start < len(line) and (line[value_start] == ' ' or line[value_start] == ':'):
                value_start += 1
            var value_end = value_start
            while value_end < len(line) and line[value_end] >= '0' and line[value_end] <= '9':
                value_end += 1
            if value_end > value_start:
                return atol(String(line[value_start:value_end]))
    return 0

@fieldwise_init
struct NumaNode(Copyable, Movable):
    var id: Int
    var cpu_ids: List[Int]
    var distances: List[Int]
    var mem_total_kb: Int
    var mem_free_kb: Int

    fn __init__(out self, id: Int):
        self.id = id
        self.cpu_ids = List[Int]()
        self.distances = List[Int]()
        self.mem_total_kb = 0
        self.mem_free_kb = 0

struct NumaInfo:
    var nodes: List[NumaNode]
    var num_nodes: Int

    fn __init__(out self):
        self.num_nodes = 0
        self.nodes = List[NumaNode]()
        try:
            var online_str = read_sysfs("/sys/devices/system/node/online")
            if len(online_str) == 0:
                return
            var node_ids = parse_cpulist(online_str)
            for i in range(len(node_ids)):
                var node_id = node_ids[i]
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

    fn get_node_cpus(self, node_id: Int) -> List[Int]:
        """Get the list of CPU IDs belonging to the specified NUMA node."""
        if node_id < 0 or node_id >= self.num_nodes:
            return List[Int]()
        return self.nodes[node_id].cpu_ids.copy()

    fn get_node_mask[mask_size: Int = 128](self, node_id: Int) -> CpuMask[mask_size]:
        var mask = CpuMask[mask_size]()
        if node_id < 0 or node_id >= self.num_nodes:
            return mask^
        var cpus = self.nodes[node_id].cpu_ids.copy()
        for i in range(len(cpus)):
            mask.set(cpus[i])
        return mask^

    fn distance(self, from_node: Int, to_node: Int) -> Int:
        """Get the NUMA distance between two nodes."""
        if from_node < 0 or from_node >= self.num_nodes:
            return -1
        if to_node < 0 or to_node >= len(self.nodes[from_node].distances):
            return -1
        return self.nodes[from_node].distances[to_node]

    fn cpus_per_node(self) -> Int:
        """Get the number of CPUs per node (assumes uniform topology)."""
        if self.num_nodes == 0:
            return 0
        return len(self.nodes[0].cpu_ids)

    fn cpus_on_node(self, node: Int) -> Int:
        """Get the number of CPUs on a specific node."""
        if node < 0 or node >= self.num_nodes:
            return 0
        return len(self.nodes[node].cpu_ids)

    fn total_cpus(self) -> Int:
        """Get the total number of CPUs across all nodes."""
        var total = 0
        for i in range(self.num_nodes):
            total += len(self.nodes[i].cpu_ids)
        return total

    fn print_debug(self):
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
