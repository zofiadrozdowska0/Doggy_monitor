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


abstract class MonitoringFragment : Fragment() {




    private var viewer: MjpegView? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?


    ): View {

        val view = inflater.inflate(R.layout.fragment_monitoring, container, false)
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





    override fun onDestroyView() {
        super.onDestroyView()
        viewer?.stopStream()
        viewer = null
    }
}