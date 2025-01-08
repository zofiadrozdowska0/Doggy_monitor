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
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.io.PrintWriter


class MonitoringFragment : Fragment() {


//    private var _binding: FragmentMonitoringBinding? = null
//    private val binding get() = _binding!!
//
//    private var player: ExoPlayer? = null

    private var _binding: FragmentMonitoringBinding? = null
    private val binding get() = _binding!!

    private var viewer: MjpegView? = null
    private var emotionTextView: TextView? = null
    private var emotionContainer: androidx.cardview.widget.CardView? = null

    private var serverIp = "192.168.137.1" // Adres serwera
    private var serverPort = 5005           // Port serwera
    private lateinit var textView: TextView
    private var isRunning = true




    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMonitoringBinding.inflate(inflater, container, false)
        val root: View = binding.root

        // Odwołaj się do elementów UI za pomocą binding
        emotionTextView = binding.DogEmotion
        emotionContainer = binding.DogEmotionCont

        startEmotionMonitoring()

        viewer = binding.mjpegview
        viewer?.apply {
            setMode(MjpegView.MODE_FIT_WIDTH)
            setAdjustHeight(true)
            setSupportPinchZoomAndPan(true)
            setUrl(SettingsPreferences.server_address)
            startStream()
        }

        return root
    }

    private fun startEmotionMonitoring() {
        CoroutineScope(Dispatchers.IO).launch {
            var socket: Socket? = null
            var reader: BufferedReader? = null
            var writer: PrintWriter? = null

            try {
                // Nawiąż połączenie z serwerem
                socket = Socket(serverIp, serverPort)
                reader = BufferedReader(InputStreamReader(socket.getInputStream()))
                writer = PrintWriter(socket.getOutputStream(), true)

                // Wyślij wiadomość inicjującą
                writer.println("start_emotion")
                Log.d("Emotion", "Start")
                isRunning = true

                // Odbieraj dane od serwera
                while (isRunning) {
                    Log.d("Emotion", "Odbieranie")
                    val message = reader.readLine() ?: break
                    Log.d("Emotion", "odebrano: $message")

                    // Zaktualizuj interfejs użytkownika
                    withContext(Dispatchers.Main) {
                        emotionTextView?.text = message ?: "Brak danych"
                    }

                }
            } catch (e: Exception) {
                Log.d("Emotion", "Błąd połączenia")
                e.printStackTrace()

                // Zaktualizuj interfejs użytkownika z błędem
                withContext(Dispatchers.Main) {
                    emotionTextView?.text = "Błąd połączenia z serwerem"
                }
            } finally {
                Log.d("Emotion", "Zakończono")
                // Zamknij zasoby
                reader?.close()
                writer?.close()
                socket?.close()
            }
        }
    }



    override fun onDestroyView() {
        super.onDestroyView()
        viewer?.stopStream()
        viewer = null

        isRunning = false
        _binding = null


    }
}