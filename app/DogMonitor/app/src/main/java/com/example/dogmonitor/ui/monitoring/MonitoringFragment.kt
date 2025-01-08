package com.example.dogmonitor.ui.monitoring

import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.example.dogmonitor.SettingsPreferences
import com.example.dogmonitor.databinding.FragmentMonitoringBinding
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import com.google.android.exoplayer2.ui.PlayerView

class MonitoringFragment : Fragment() {

    private var _binding: FragmentMonitoringBinding? = null
    private val binding get() = _binding!!

    private var player: ExoPlayer? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        // Inflate the layout for this fragment
        _binding = FragmentMonitoringBinding.inflate(inflater, container, false)
        val root: View = binding.root

        // Initialize PlayerView
        initializePlayer() // Call to initialize the player

        return root
    }

    private fun initializePlayer() {
        player = ExoPlayer.Builder(requireContext()).build()
        binding.playerView.player = player // Accessing PlayerView using View Binding

        // Set the media item to be played (replace with your video URL)
        val videoUri = Uri.parse(SettingsPreferences.server_address)
        val mediaItem = MediaItem.fromUri(videoUri)
        player?.setMediaItem(mediaItem)

        // Prepare and start playback
        player?.prepare()
        player?.playWhenReady = true
    }

    override fun onPause() {
        super.onPause()
        player?.pause() // Pause the player when the fragment is paused
    }

    override fun onStop() {
        super.onStop()
        releasePlayer() // Release the player when the fragment is stopped
    }

    private fun releasePlayer() {
        player?.release()
        player = null // Clear the player reference
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null // Clear the binding reference
    }
}