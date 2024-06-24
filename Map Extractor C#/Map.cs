using System;
using GBX.NET;
using GBX.NET.Engines.Game;

class Map {
    private CGameCtnChallenge map;
    public Map(string file_name) {
        map = GameBox.ParseNode<CGameCtnChallenge>(file_name);
    }

    public void print_blocks(){
        foreach (CGameCtnBlock block in map.GetBlocks()) {
            if (block == null)
                continue;
            String name = block.Name.Replace("TrackWall", "RoadTech").Replace("Pillar", "");

            Console.WriteLine("Block: " + name);
            Console.WriteLine("Position: " + block.Coord);
            Console.WriteLine("Rotation: " + block.Direction);
            Console.WriteLine("===================");
        }
    }

    public void export_to_file(string file_name){
        // open .txt for writing
        System.IO.StreamWriter file = new System.IO.StreamWriter(file_name);
        foreach (CGameCtnBlock block in map.GetBlocks()) {
            if (block == null)
                continue;
            if (block.Name.Contains("TrackWall")) {
                continue;
            }
            
            String name = block.Name.Replace("TrackWall", "RoadTech").Replace("Pillar", "");
            file.Write(name + ";");
            file.Write(block.Coord.X + "," + block.Coord.Y + "," + block.Coord.Z + ";");
            file.WriteLine(block.Direction);
            
        }
        file.Close();
    }

}