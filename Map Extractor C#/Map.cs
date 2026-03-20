using System;
using System.Collections.Generic;
using System.IO;
using GBX.NET;
using GBX.NET.Engines.Game;

class Map {
    private readonly CGameCtnChallenge map;

    public Map(string file_name) {
        if (!File.Exists(file_name)) {
            throw new FileNotFoundException("Map file was not found: " + Path.GetFullPath(file_name), file_name);
        }

        map = GameBox.ParseNode<CGameCtnChallenge>(file_name);
    }

    private IEnumerable<CGameCtnBlock> GetValidBlocks() {
        foreach (CGameCtnBlock block in map.GetBlocks()) {
            if (block != null) {
                yield return block;
            }
        }
    }

    public void print_blocks() {
        foreach (CGameCtnBlock block in GetValidBlocks()) {
            String name = block.Name.Replace("TrackWall", "RoadTech").Replace("Pillar", "");

            Console.WriteLine("Block: " + name);
            Console.WriteLine("Position: " + block.Coord);
            Console.WriteLine("Rotation: " + block.Direction);
            Console.WriteLine("===================");
        }
    }

    public void export_to_file(string file_name) {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(file_name))!);

        using StreamWriter file = new StreamWriter(file_name);
        foreach (CGameCtnBlock block in GetValidBlocks()) {
            if (block.Name.Contains("TrackWall")) {
                continue;
            }

            String name = block.Name.Replace("TrackWall", "RoadTech").Replace("Pillar", "");
            file.Write(name + ";");
            file.Write(block.Coord.X + "," + block.Coord.Y + "," + block.Coord.Z + ";");
            file.WriteLine(block.Direction);
        }
    }
}
