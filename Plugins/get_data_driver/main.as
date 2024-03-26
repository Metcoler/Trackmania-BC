

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
		Net::Socket@ sock_serv = Net::Socket();
		
		if (!sock_serv.Listen("127.0.0.1", 9002))
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
		print("Getting write...");
		while (!sock.CanWrite())
		{
			yield();
		}
		print("Socket can write");

		print(Time::Now + ": Connected!");
		
		// OpenPlanet can store bytes in a MemoryBuffer:
		MemoryBuffer@ buf = MemoryBuffer(0);

		// count distance
		float distance = 0;	
		float previous_x = 0;
		float previous_y = 0;
		float previous_z = 0;

		// packet series.
		int packet_number = 0;


		// connection is established, we can start streaming data:
		bool cc = true;
		while(cc)
		{
			// We check whether the player API is reachable
			// In case it is not, we yield to check back later

			CTrackMania@ app = cast<CTrackMania>(GetApp());
			
			if(app is null)
			{
				print("Waiting to open app...");
				packet_number = 0;
				yield();
				continue;
			}
			
			CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
			if(playground is null)
			{
				print("Waiting to open playground...");
				packet_number = 0;
				yield();
				continue;
			}
			
			CSmArena@ arena = cast<CSmArena>(playground.Arena);
			if(arena is null)
			{
				print("Waiting to open arena...");
				packet_number = 0;
				yield();
				continue;
			}
			
			if(arena.Players.Length <= 0)
			{
				print("Waiting for players...");
				packet_number = 0;
				yield();
				continue;
			}

			CSmPlayer@ player = arena.Players[0];
			if(player is null)
			{
				print("Getting player 0...");
				packet_number = 0;
				yield();
				continue;
			}
	

			CSceneVehicleVis@ vis = VehicleState::GetVis(GetApp().GameScene, player);
			if(vis is null)
			{
				print("Waiting to get vis player ...");
				packet_number = 0;
				yield();
				continue;
			}

			CSceneVehicleVisState@ state = vis.AsyncState;
			if(state is null)
			{
				print("Waiting to get viewing player state...");
				packet_number = 0;
				yield();
				continue;
			}
			auto race_state = playground.GameTerminals[0].UISequence_Current;

			if (race_state == SGamePlaygroundUIConfig::EUISequence::EndRound) {
				print("Waiting for player start...");
				packet_number = 0;
				yield();				
				continue;
			}

	
		
			// The state is ready.
			// We can send data to TMRL for this TrackMania frame:
			print("Sending data: " + packet_number++);
			
			// distance update
			if (previous_x == 0 && previous_y == 0 && previous_z == 0) {
				previous_x = state.Position.x;
				previous_y = state.Position.y;
				previous_z = state.Position.z;
			}

			distance += Math::Sqrt(
				(state.Position.x - previous_x) * (state.Position.x - previous_x) +
				(state.Position.y - previous_y) * (state.Position.y - previous_y) +
				(state.Position.z - previous_z) * (state.Position.z - previous_z)
			);
			previous_x = state.Position.x;
			previous_y = state.Position.y;
			previous_z = state.Position.z;
		
			// get player time 
			float time = (GetApp().PlaygroundScript.Now - player.StartTime) / 1000.0;

			// TODO detect car crashing 	
			
		

			// place cursor at the beginning of the buffer to erase previous data:
			buf.Seek(0, 0);
			
			// write data to the buffer (the plugin sends everything as floats):
			
			//buf.Write(api.Speed);
			append_float(buf, state.FrontSpeed);

			append_float(buf, VehicleState::GetSideSpeed(state));
			
			//buf.Write(api.Distance);
			append_float(buf, distance);
			
			//buf.Write(api.Position.x);
			append_float(buf, state.Position.x);
			
			//buf.Write(api.Position.y);
			append_float(buf, state.Position.y);
			
			//buf.Write(api.Position.z);
			append_float(buf, state.Position.z);
			
			//buf.Write(api.InputSteer);
			append_float(buf, state.InputSteer);
			
			//buf.Write(api.InputGasPedal);
			append_float(buf, state.InputGasPedal);
			
			
			//if(api.InputIsBraking) buf.Write(1.0f);
			//else buf.Write(0.0f);
			if(state.InputIsBraking) append_float(buf, 1.0f);
			else append_float(buf, 0.0f);
			
			// can use CGamePlaygroundUIConfig::EUISequence::Finish or CGameTerminal::ESGamePlaygroundUIConfig__EUISequence::Finish
			
			if (race_state == CGamePlaygroundUIConfig::EUISequence::Finish) buf.Write(1.0f);
            else buf.Write(0.0f);

			
			//buf.Write(api.EngineCurGear);
			append_float(buf, state.CurGear);
			
			//buf.Write(api.EngineRpm);
			append_float(buf, VehicleState::GetRPM(state));

			//buf.Write(api.AimDirection.x)
			append_float(buf, state.Dir.x);
			
			//buf.Write(api.AimDirection.y)
			append_float(buf, state.Dir.y);
			
			//buf.Write(api.AimDirection.z)
			append_float(buf, state.Dir.z);

			//buf.Write(api.time)
			append_float(buf, time);




			
			buf.Seek(0, 0);
			
			cc = send_memory_buffer(sock, buf);


			yield();  // this statement stops the script until the next frame
		}
		sock.Close();
		sock_serv.Close();
	}
}
