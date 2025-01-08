package com.example.dogmonitor.ui.monitoring

import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import androidx.fragment.app.Fragment
import com.example.dogmonitor.R
import com.example.dogmonitor.SettingsPreferences
import com.example.dogmonitor.databinding.FragmentMonitoringBinding
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import com.google.android.exoplayer2.ui.PlayerView
import com.longdo.mjpegviewer.MjpegView


class MonitoringFragment : Fragment() {

//    private var _binding: FragmentMonitoringBinding? = null
//    private val binding get() = _binding!!
//
//    private var player: ExoPlayer? = null


    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?


    ): View {
        // Inflate the layout for this fragment
//        _binding = FragmentMonitoringBinding.inflate(inflater, container, false)
//        val root: View = binding.root
        val view = inflater.inflate(R.layout.fragment_monitoring, container, false)
        val viewer = view.findViewById<MjpegView>(R.id.mjpegview)
        viewer.setMode(MjpegView.MODE_FIT_WIDTH);
        viewer.setAdjustHeight(true);
        viewer.setSupportPinchZoomAndPan(true);
        viewer.setUrl("http://192.168.137.182:5000/video_feed");
        viewer.startStream();
        // Initialize PlayerView
//        initializePlayer() // Call to initialize the player

        return view
    }





    override fun onDestroyView() {
        super.onDestroyView()

    }
}