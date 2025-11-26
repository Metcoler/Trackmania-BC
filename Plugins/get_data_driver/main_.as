// This plugin ships with the TMRL framework.
// It sends game data to the default TrackMania Gym environments.

// send the content of buf over socket sock:
bool send_memory_buffer(Net::Socket@ sock, MemoryBuffer@ buf)
{
	if (!sock.Write(buf))
	{
		// If this fails, the socket might not be open. Something is wrong!
		print("INFO: Disconnected, could not send data.");
		return false;
	}
	return true;
}

// cast val to a float when necessary and append it to buf:
void append_float(MemoryBuffer@ buf, float val)
{
	buf.Write(val);
}

// entry point:
void Main()
{
	while(true)
	{
		// open localhost TCP connection on port 9000:
		auto sock_serv = Net::Socket();
		if (!sock_serv.Listen("127.0.0.1", 9000))
		{
			print("Could not initiate server socket.");
			return;
		}
		print(Time::Now + ": Waiting for incoming connection...");

		while(!sock_serv.CanRead())
		{
			yield();
		}
		print("Socket can read");
		auto sock = sock_serv.Accept();

		print(Time::Now + ": Accepted incoming connection.");

		while (!sock.CanWrite())
		{
			yield();
		}
		print("Socket can write");
		print(Time::Now + ": Connected!");
		
		// OpenPlanet can store bytes in a MemoryBuffer:
		MemoryBuffer@ buf = MemoryBuffer(0);
		
		// connection is established, we can start streaming data:
		bool cc = true;
		while(cc)
		{
			// We check whether the player API is reachable
			// In case it is not, we yield to check back later

			CTrackMania@ app = cast<CTrackMania>(GetApp());
			if(app is null)
			{
				yield();
				continue;
			}
			
			CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
			if(playground is null)
			{
				yield();
				continue;
			}
			
			CSmArena@ arena = cast<CSmArena>(playground.Arena);
			if(arena is null)
			{
				yield();
				continue;
			}
			
			if(arena.Players.Length <= 0)
			{
				yield();
				continue;
			}
			
			auto player = arena.Players[0];
			if(player is null)
			{
				yield();
				continue;
			}

			CSmScriptPlayer@ api = cast<CSmScriptPlayer>(player.ScriptAPI);
			if(api is null)
			{
				yield();
				continue;
			}
			
			// The player API is ready.
			// We can send data to TMRL for this TrackMania frame:

			auto race_state = playground.GameTerminals[0].UISequence_Current;
			
			// place cursor at the beginning of the buffer to erase previous data:
			buf.Seek(0, 0);
			
			// write data to the buffer (the plugin sends everything as floats):
			//buf.Write(api.Speed);
			append_float(buf, api.Speed);
			
			//buf.Write(api.Distance);
			append_float(buf, api.Distance);
			
			//buf.Write(api.Position.x);
			append_float(buf, api.Position.x);
			
			//buf.Write(api.Position.y);
			append_float(buf, api.Position.y);
			
			//buf.Write(api.Position.z);
			append_float(buf, api.Position.z);
			
			//buf.Write(api.InputSteer);
			append_float(buf, api.InputSteer);
			
			//buf.Write(api.InputGasPedal);
			append_float(buf, api.InputGasPedal);
			
			//if(api.InputIsBraking) buf.Write(1.0f);
			//else buf.Write(0.0f);
			if(api.InputIsBraking) append_float(buf, 1.0f);
			else append_float(buf, 0.0f);
			
			// can use CGamePlaygroundUIConfig::EUISequence::Finish or CGameTerminal::ESGamePlaygroundUIConfig__EUISequence::Finish
			
			//if(race_state == SGamePlaygroundUIConfig::EUISequence::Finish) buf.Write(1.0f);
            //else buf.Write(0.0f);
			if(race_state == SGamePlaygroundUIConfig::EUISequence::Finish) append_float(buf, 1.0f);
            else append_float(buf, 0.0f);
			
			//buf.Write(api.EngineCurGear);
			append_float(buf, api.EngineCurGear);
			
			//buf.Write(api.EngineRpm);
			append_float(buf, api.EngineRpm);
			
			buf.Seek(0, 0);
			
			cc = send_memory_buffer(sock, buf);

			yield();  // this statement stops the script until the next frame
		}
		sock.Close();
		sock_serv.Close();
	}
}
