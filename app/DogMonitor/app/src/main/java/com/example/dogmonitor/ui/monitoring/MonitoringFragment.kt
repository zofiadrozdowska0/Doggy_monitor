package com.example.dogmonitor.ui.monitoring

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.example.dogmonitor.R

import com.example.dogmonitor.SettingsPreferences
import com.example.dogmonitor.databinding.FragmentMonitoringBinding
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import com.google.android.exoplayer2.ui.PlayerView
import com.longdo.mjpegviewer.MjpegView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.Socket


class MonitoringFragment : Fragment() {


//    private var _binding: FragmentMonitoringBinding? = null
//    private val binding get() = _binding!!
//
//    private var player: ExoPlayer? = null


    private var viewer: MjpegView? = null
    private var emotionTextView: TextView? = null
    private var emotionContainer: androidx.cardview.widget.CardView? = null

    private var job: Job? = null
    private val port = 8488
    private var socket: DatagramSocket? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?


    ): View {

        val view = inflater.inflate(R.layout.fragment_monitoring, container, false)

        emotionTextView = view.findViewById<TextView>(R.id.DogEmotion)
        emotionContainer = view.findViewById<androidx.cardview.widget.CardView>(R.id.DogEmotionCont)

        startBroadcastReceiver()


        viewer = view.findViewById<MjpegView>(R.id.mjpegview)
        viewer?.apply {
            setMode(MjpegView.MODE_FIT_WIDTH)
            setAdjustHeight(true)
            setSupportPinchZoomAndPan(true)
            setUrl(SettingsPreferences.server_address)
            startStream()
        }


        return view
    }

    private fun startBroadcastReceiver() {
        Log.d("MonitoringFragment", "Uruchamianie nasłuchiwania na porcie $port")

        job = CoroutineScope(Dispatchers.IO).launch {

            try {
                socket = DatagramSocket(port)
                socket?.broadcast = true

                while (true) {
                    Log.d("MonitoringFragment", "1")
                    val buffer = ByteArray(1024)
                    Log.d("MonitoringFragment", "2")
                    val packet = DatagramPacket(buffer, buffer.size)
                    Log.d("MonitoringFragment", "3")
                    socket?.receive(packet)
                    Log.d("MonitoringFragment", "4")

                    val receivedText = String(packet.data, 0, packet.length)
                    withContext(Dispatchers.Main) {
                        emotionTextView?.text = receivedText
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    emotionTextView?.text = "Błąd odbioru: ${e.message}"
                }
            } finally {
                socket?.close() // Zamknięcie socketu
            }
        }
    }



    override fun onDestroyView() {
        super.onDestroyView()
        viewer?.stopStream()
        viewer = null
        job?.cancel()
        socket?.close()

    }
}