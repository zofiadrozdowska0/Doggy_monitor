package com.example.dogmonitor.ui.setting

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.Switch
import android.widget.TextView
import androidx.core.text.isDigitsOnly
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.dogmonitor.R
import com.example.dogmonitor.SettingsPreferences
import com.example.dogmonitor.databinding.FragmentSettingBinding

class SettingFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_setting, container, false)

        // Initialize settings from SharedPreferences
        val serverAddressEditText = view.findViewById<EditText>(R.id.server_address)
        val notificationsSwitch = view.findViewById<Switch>(R.id.switch_notifications)
        val videoClipLengthEditText = view.findViewById<EditText>(R.id.video_clip_lenght)

        val notificationHappyCheckBox = view.findViewById<CheckBox>(R.id.notification_indicator_happy_checkBox)
        val notificationSadCheckBox = view.findViewById<CheckBox>(R.id.notification_indicator_sad_checkBox)
        val notificationAngryCheckBox = view.findViewById<CheckBox>(R.id.notification_indicator_angry_checkBox)
        val notificationHungryCheckBox = view.findViewById<CheckBox>(R.id.notification_indicator_hungry_checkBox)

        val videoClipHappyCheckBox = view.findViewById<CheckBox>(R.id.video_clip_indicator_happy_checkBox)
        val videoClipSadCheckBox = view.findViewById<CheckBox>(R.id.video_clip_indicator_sad_checkBox)
        val videoClipAngryCheckBox = view.findViewById<CheckBox>(R.id.video_clip_indicator_angry_checkBox)
        val videoClipHungryCheckBox = view.findViewById<CheckBox>(R.id.video_clip_indicator_hungry_checkBox)

        // Load values from SettingsPreferences
        serverAddressEditText.setText(SettingsPreferences.server_address)
        notificationsSwitch.isChecked = SettingsPreferences.notificationsEnabled
        videoClipLengthEditText.setText(SettingsPreferences.videoclipLenght.toString())

        notificationHappyCheckBox.isChecked = SettingsPreferences.notificationsIndicatorHappy
        notificationSadCheckBox.isChecked = SettingsPreferences.notificationsIndicatorSad
        notificationAngryCheckBox.isChecked = SettingsPreferences.notificationsIndicatorAngry
        notificationHungryCheckBox.isChecked = SettingsPreferences.notificationsIndicatorHungry

        videoClipHappyCheckBox.isChecked = SettingsPreferences.videoclipIndicatorHappy
        videoClipSadCheckBox.isChecked = SettingsPreferences.videoclipIndicatorSad
        videoClipAngryCheckBox.isChecked = SettingsPreferences.videoclipIndicatorAngry
        videoClipHungryCheckBox.isChecked = SettingsPreferences.videoclipIndicatorHungry

        // Set listeners to save changes
        notificationsSwitch.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.notificationsEnabled = isChecked
        }

        serverAddressEditText.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                SettingsPreferences.server_address = serverAddressEditText.text.toString()
            }
        }

        videoClipLengthEditText.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                SettingsPreferences.videoclipLenght = videoClipLengthEditText.text.toString().toIntOrNull() ?: 0
            }
        }

        notificationHappyCheckBox.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.notificationsIndicatorHappy = isChecked
        }
        notificationSadCheckBox.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.notificationsIndicatorSad = isChecked
        }
        notificationAngryCheckBox.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.notificationsIndicatorAngry = isChecked
        }
        notificationHungryCheckBox.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.notificationsIndicatorHungry = isChecked
        }

        videoClipHappyCheckBox.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.videoclipIndicatorHappy = isChecked
        }
        videoClipSadCheckBox.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.videoclipIndicatorSad = isChecked
        }
        videoClipAngryCheckBox.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.videoclipIndicatorAngry = isChecked
        }
        videoClipHungryCheckBox.setOnCheckedChangeListener { _, isChecked ->
            SettingsPreferences.videoclipIndicatorHungry = isChecked
        }

        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Find the switch and checkboxes by their IDs

        val notificationsSwitch = view.findViewById<Switch>(R.id.switch_notifications)
        val videoClipLengthEditText = view.findViewById<EditText>(R.id.video_clip_lenght)


        val notificationsOptions = view.findViewById<androidx.cardview.widget.CardView>(R.id.notifications_options)
        val videoClipOptions = view.findViewById<androidx.cardview.widget.CardView>(R.id.video_clip_options)

        if (notificationsSwitch.isChecked()){
            notificationsOptions.visibility = View.VISIBLE

        } else{
            notificationsOptions.visibility = View.GONE
        }

        if(videoClipLengthEditText.text.toString().toIntOrNull() != 0 and videoClipLengthEditText.text.toString().toIntOrNull()!! != null) {

            videoClipOptions.visibility = View.VISIBLE

            }
        else{
            videoClipOptions.visibility = View.GONE
        }

        videoClipLengthEditText.setOnFocusChangeListener{ _, hasFocus ->
            if (!hasFocus) {
                if(videoClipLengthEditText.text.toString().toIntOrNull() != 0 and videoClipLengthEditText.text.toString().toIntOrNull()!! != null) {

                    videoClipOptions.visibility = View.VISIBLE

                }
                else{
                    videoClipOptions.visibility = View.GONE
                }
            }
        }




        // Toggle the visibility of checkboxes based on the switch state
        notificationsSwitch.setOnCheckedChangeListener { _, isChecked ->
            val visibility = if (isChecked) View.VISIBLE else View.GONE
            notificationsOptions.visibility = visibility

        }
    }
}