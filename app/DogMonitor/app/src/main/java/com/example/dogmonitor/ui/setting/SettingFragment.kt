package com.example.dogmonitor.ui.setting

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.RadioButton
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


        val dogBreed1RadioBox = view.findViewById<RadioButton>(R.id.Breed1)
        val dogBreed2RadioBox = view.findViewById<RadioButton>(R.id.Breed2)
        val dogBreed3RadioBox = view.findViewById<RadioButton>(R.id.Breed3)


        // Load values from SettingsPreferences
        serverAddressEditText.setText(SettingsPreferences.server_address)


        serverAddressEditText.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                SettingsPreferences.server_address = serverAddressEditText.text.toString()
            }
        }



        if (dogBreed1RadioBox.isChecked){
            SettingsPreferences.dog_breed = 1
        }
        else if (dogBreed2RadioBox.isChecked){
            SettingsPreferences.dog_breed = 2
        }
        else if (dogBreed2RadioBox.isChecked){
            SettingsPreferences.dog_breed = 3
        }
        else {
            SettingsPreferences.dog_breed = 0
        }

        return view
    }


}