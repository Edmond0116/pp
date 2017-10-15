#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

int main(int argc, char **argv) {
    ifstream ifs(argv[1]);
    stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();
    string type;
    float tmp, io = 0, comm = 0, total = 0;
    int iot = 0, commt = 0;
    while (ss >> type >> tmp) {
        if (type == "Total:") total = max(total, tmp);
        else if (type == "IO:") io += tmp, ++iot;
        else if (type ==  "Comm:") comm += tmp, ++commt;
        else cout << "[Unknown] " << type << " " << tmp << endl;
    }
    float io_avg = 2 * io / iot,
        comm_avg = comm / commt,
        comp = total - io_avg - comm_avg;
    cout << "(Comm, IO, Comp, Total) =" << endl <<
        comm_avg << "," <<
        io_avg  << "," <<
        comp << "," <<
        total << endl;
    return 0;
}
