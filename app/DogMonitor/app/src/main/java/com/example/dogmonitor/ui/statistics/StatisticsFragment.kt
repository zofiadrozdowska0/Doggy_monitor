package com.example.dogmonitor.ui.statistics

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.dogmonitor.R
import com.example.dogmonitor.databinding.FragmentStatisticsBinding
import com.google.android.material.internal.ViewUtils.dpToPx
import java.time.LocalDateTime
import java.util.Calendar
import android.util.Log

class StatisticsFragment : Fragment() {

    private var _binding: FragmentStatisticsBinding? = null
    private val binding get() = _binding!!
    private val calendar = Calendar.getInstance()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_statistics, container, false)
        _binding = FragmentStatisticsBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val hourNow = calendar.get(Calendar.HOUR_OF_DAY)
        val minutesNow = calendar.get(Calendar.MINUTE)

        val timelineView = view.findViewById<LinearLayout>(R.id.time_line)

        var hoursShift = 0
        for (i in 0 until timelineView.childCount) {
            val child = timelineView.getChildAt(i)
            if (child is androidx.cardview.widget.CardView) {
                Log.e("Tag", hoursShift.toString())

                val textView = child.getChildAt(0) as? TextView
                textView?.let {
                    if (hoursShift == 0) {
                        // For the first entry (current time)
                        val height = (60 - minutesNow) * 2
                        child.layoutParams.height = height
                        it.text = "$hourNow:$minutesNow"
                    } else {
                        if (hoursShift == 23) {
                            // For the last entry (current hour, but no minutes)
                            val height = minutesNow * 2
                            child.layoutParams.height = height
                            it.text = "$hourNow:00"
                        } else {
                            // For all other hours
                            val height = 120
                            child.layoutParams.height = height
                            it.text = "$hourNow:00"
                        }
                    }

                    // Request layout update after modifying height and text
                    child.requestLayout()
                    Log.e("Tag", it.text.toString()) // Log the text for debugging
                }

                hoursShift++
            }
        }

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}