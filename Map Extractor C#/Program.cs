using System;
using System.IO;

class Program {
    private const string DefaultMapName = "pallete";

    static int Main(string[] args) {
        string repoRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../"));
        string mapsDir = Path.Combine(repoRoot, "Maps", "GameFiles");
        string exportsDir = Path.Combine(repoRoot, "Maps", "ExportedBlocks");

        if (args.Length > 0 && IsHelpFlag(args[0])) {
            PrintUsage(mapsDir, exportsDir);
            return 0;
        }

        string mapInput = args.Length > 0 ? args[0] : DefaultMapName;
        string mapPath = ResolveMapPath(mapInput, mapsDir);
        string exportPath = args.Length > 1
            ? Path.GetFullPath(args[1])
            : Path.Combine(exportsDir, GetExportFileName(mapPath));

        try {
            Console.WriteLine("Map Extractor");
            Console.WriteLine("Map input : " + mapInput);
            Console.WriteLine("Map file  : " + mapPath);
            Console.WriteLine("Export to : " + exportPath);

            Map map = new Map(mapPath);
            map.print_blocks();
            map.export_to_file(exportPath);
            Console.WriteLine("Exported blocks to: " + exportPath);
            return 0;
        }
        catch (FileNotFoundException ex) {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
        catch (EndOfStreamException ex) {
            Console.Error.WriteLine("GBX.NET could not fully parse this map.");
            Console.Error.WriteLine("This usually means the .Map.Gbx file is newer than the installed GBX.NET version, or the file is truncated/corrupted.");
            Console.Error.WriteLine("File: " + mapPath);
            Console.Error.WriteLine("Details: " + ex.Message);
            return 1;
        }
        catch (Exception ex) {
            Console.Error.WriteLine("Failed to read map '" + mapPath + "'.");
            Console.Error.WriteLine(ex.ToString());
            return 1;
        }
    }

    private static bool IsHelpFlag(string value) {
        return value.Equals("-h", StringComparison.OrdinalIgnoreCase)
            || value.Equals("--help", StringComparison.OrdinalIgnoreCase)
            || value.Equals("/?", StringComparison.OrdinalIgnoreCase);
    }

    private static void PrintUsage(string mapsDir, string exportsDir) {
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project \"Map Extractor C#/C# test.csproj\" -- [map-name-or-path] [output-path]");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  dotnet run --project \"Map Extractor C#/C# test.csproj\"");
        Console.WriteLine("  dotnet run --project \"Map Extractor C#/C# test.csproj\" -- \"AI Training #5\"");
        Console.WriteLine("  dotnet run --project \"Map Extractor C#/C# test.csproj\" -- \"Maps/GameFiles/AI Training #5.Map.Gbx\"");
        Console.WriteLine();
        Console.WriteLine("Defaults:");
        Console.WriteLine("  default map name : " + DefaultMapName);
        Console.WriteLine("  maps directory   : " + mapsDir);
        Console.WriteLine("  exports directory: " + exportsDir);
    }

    private static string ResolveMapPath(string mapInput, string mapsDir) {
        if (File.Exists(mapInput)) {
            return Path.GetFullPath(mapInput);
        }

        if (Path.HasExtension(mapInput)) {
            return Path.GetFullPath(mapInput);
        }

        return Path.Combine(mapsDir, mapInput + ".Map.Gbx");
    }

    private static string GetExportFileName(string mapPath) {
        string fileName = Path.GetFileNameWithoutExtension(mapPath);

        if (fileName.EndsWith(".Map", StringComparison.OrdinalIgnoreCase)) {
            fileName = Path.GetFileNameWithoutExtension(fileName);
        }

        return fileName + ".txt";
    }
}
