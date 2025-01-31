#include <iostream>
#include <vector>
#include "exodusII.h"

int main() {
    std::cout << "Reading in an example case\n";

    const char* filename = "case13_out.e"; // Path to the Exodus file
    
    float version = 0.0;
    int   appWordSize  = 0;
    int   diskWordSize = 8;
    // Open the Exodus file (read-only)
    int exoid = ex_open(filename, EX_READ, &appWordSize, &diskWordSize, &version);
    if (exoid < 0) {
        std::cerr << "Error opening Exodus file: " << filename << std::endl;
        return -1;
    }

    // Read metadata
    char title[MAX_LINE_LENGTH];
    int num_dim, num_nodes, num_elems, num_blocks, num_node_sets, num_side_sets;
    ex_get_init(exoid, title, &num_dim, &num_nodes, &num_elems, &num_blocks, &num_node_sets, &num_side_sets);

    std::cout << "Title: " << title << std::endl;
    std::cout << "Dimensions: " << num_dim << std::endl;
    std::cout << "Nodes: " << num_nodes << std::endl;
    std::cout << "Elements: " << num_elems << std::endl;
    std::cout << "Element Blocks: " << num_blocks << std::endl;

    // Read node coordinates
    std::vector<float> x(num_nodes), y(num_nodes), z(num_nodes);
    ex_get_coord(exoid, x.data(), y.data(), z.data());

    // Print first 5 nodes
    std::cout << "First 5 node coordinates:" << std::endl;
    for (int i = 0; i < std::min(5, num_nodes); i++) {
        std::cout << "Node " << i + 1 << ": (" << x[i] << ", " << y[i];
        if (num_dim == 3) {
            std::cout << ", " << z[i];
        }
        std::cout << ")" << std::endl;
    }

    // Close file
    // ex_close(exoid);
    return 0;
}