using System;

class Program {
    static void Main(string[] args) {
        //string file_name = args[0];
        string file_name = "../../../../Maps/GameFiles/small_map.Map.Gbx";
        Map map = new Map(file_name);
        map.print_blocks();
        map.export_to_file("../../../../Maps/ExportedBlocks/small_map.txt");

    }
}