using System;

class Program {
    static void Main(string[] args) {
        //string file_name = args[0];
        string map_name = "loop_test";
        string file_name = "../../../../Maps/GameFiles/" + map_name + ".Map.Gbx";
        string export_name = "../../../../Maps/ExportedBlocks/" + map_name + ".txt";
        Map map = new Map(file_name);
        map.print_blocks();
        map.export_to_file(export_name);

    }
}