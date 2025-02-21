#include <vector>
#include <iostream>
#include <string>
#include "exodusII.h"
//  #include "rtweekend.h"
// #include "vec3.h"

// A class for a single node (vertex)
struct Node {
    double x, y, z;
};

// A class for an element (tetrahedron, hexahedron, etc.)
struct Element {
    std::vector<int> nodeIndices;  // Node IDs (1-based indexing from Exodus)
};

// The main Mesh class
class Mesh {
public:
    std::vector<Node> nodes;
    std::vector<Element> elements;

    void printInfo() const {
        std::cout << "Mesh has " << nodes.size() << " nodes and " << elements.size() << " elements.\n";
    }
};

Mesh readExodusMesh(const std::string& filename) {
    Mesh mesh;

    // Open the Exodus file
    int CPU_word_size = 0, IO_word_size = 8;
    float version = 0.0;
    int exoid = ex_open(filename.c_str(), EX_READ, &CPU_word_size, &IO_word_size, &version);
    if (exoid < 0) {
        std::cerr << "Error: Failed to open Exodus file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read metadata
    char title[MAX_LINE_LENGTH + 1];
    int num_dim, num_nodes, num_elems, num_elem_blk, num_node_sets, num_side_sets;
    ex_get_init(exoid, title, &num_dim, &num_nodes, &num_elems, &num_elem_blk, &num_node_sets, &num_side_sets);

    std::cout << "Mesh Title: " << title << "\n"
              << "Dimensions: " << num_dim << "\n"
              << "Nodes: " << num_nodes << "\n"
              << "Elements: " << num_elems << "\n"
              << "Element Blocks: " << num_elem_blk << std::endl;

    // Read node coordinates
    std::vector<double> x(num_nodes), y(num_nodes), z(num_nodes);
    ex_get_coord(exoid, x.data(), y.data(), (num_dim == 3) ? z.data() : nullptr);

    // Store nodes in the mesh
    for (int i = 0; i < num_nodes; ++i) {
        mesh.nodes.push_back({x[i], y[i], z[i]});
    }

    // Read element block IDs
    std::vector<int> elem_blk_ids(num_elem_blk);
    ex_get_ids(exoid, EX_ELEM_BLOCK, elem_blk_ids.data());

    // Read element connectivity
    for (int blk = 0; blk < num_elem_blk; ++blk) {
        int blk_id = elem_blk_ids[blk];
        char elem_type[MAX_STR_LENGTH + 1];
        int num_elems_in_blk, num_nodes_per_elem, num_attrs;

        ex_get_block(exoid, EX_ELEM_BLOCK, blk_id, elem_type, &num_elems_in_blk, &num_nodes_per_elem, nullptr, nullptr, &num_attrs);

        std::cout << "Block " << blk_id << " (" << elem_type << ") has " << num_elems_in_blk << " elements, each with " << num_nodes_per_elem << " nodes." << std::endl;

        // Read connectivity
        std::vector<int> connectivity(num_elems_in_blk * num_nodes_per_elem);
        ex_get_conn(exoid, EX_ELEM_BLOCK, blk_id, connectivity.data(), nullptr, nullptr);

        // Convert Exodus node IDs (1-based) to 0-based
        for (int i = 0; i < num_elems_in_blk; ++i) {
            Element elem;
            for (int j = 0; j < num_nodes_per_elem; ++j) {
                elem.nodeIndices.push_back(connectivity[i * num_nodes_per_elem + j] - 1);  // Convert to 0-based indexing
            }
            mesh.elements.push_back(elem);
        }
    }

    // Close the file
    ex_close(exoid);

    return mesh;
}


int main() {
    std::string filename = "case13_out.e";

    // Read and construct the mesh
    Mesh mesh = readExodusMesh(filename);

    // Print mesh summary
    mesh.printInfo();

    // Print some node and element data
    std::cout << "First Node: (" << mesh.nodes[0].x << ", " 
              << mesh.nodes[0].y << ", " << mesh.nodes[0].z << ")\n";

    if (!mesh.elements.empty()) {
        std::cout << "First Element Connectivity: ";
        for (int idx : mesh.elements[0].nodeIndices) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}