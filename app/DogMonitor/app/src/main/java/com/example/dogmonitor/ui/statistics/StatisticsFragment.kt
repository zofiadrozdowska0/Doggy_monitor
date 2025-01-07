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
import androidx.cardview.widget.CardView

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



        return root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val hoursContainer = view.findViewById<LinearLayout>(R.id.time_line)
        val currentTime = Calendar.getInstance()

        val currentHour = currentTime.get(Calendar.HOUR_OF_DAY)
        val currentMinute = currentTime.get(Calendar.MINUTE)

        val boxHeightPx = 300 // Wysokość każdej karty w px
        val remainingMinutes = 60 - currentMinute

        // Wyliczamy dynamiczną wysokość dla aktualnej godziny
        val currentCardHeightPx = (remainingMinutes / 60f * boxHeightPx).toInt()

        // Dodajemy dynamiczną kartę dla aktualnej godziny
        if (remainingMinutes != 0) {
            val currentCard = createCardWithTextView("$currentHour:00", currentCardHeightPx)
            hoursContainer.addView(currentCard)
        }

        // Dodajemy pozostałe godziny
        for (iter in  1..23) {
            var hour = 0;
            if (currentHour + iter > 23){
                hour = currentHour + iter - 24;
            }
            else{
                hour = currentHour+iter;
            }
            val hourCard = createCardWithTextView("$hour:00", boxHeightPx)
            hoursContainer.addView(hourCard)
        }
        val lastCardHeightPx = (currentMinute / 60f * boxHeightPx).toInt()
        if (currentMinute != 0) {
            val lastCard = createCardWithTextView("$currentHour:00", lastCardHeightPx)
            hoursContainer.addView(lastCard)
        }
    }

    private fun createCardWithTextView(timeText: String, cardHeightPx: Int): CardView {
        // Tworzymy CardView
        val cardView = CardView(requireContext()).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                cardHeightPx
            ).apply {
                bottomMargin = 16 // Margines między kolejnymi CardView
            }
            setCardBackgroundColor(resources.getColor(R.color.teal_700, requireContext().theme))
            radius = 10f // Zaokrąglenie narożników
            cardElevation = 6f // Cień karty
            textAlignment = View.TEXT_ALIGNMENT_CENTER // Centrowanie tekstu
        }

        // Tworzymy TextView do umieszczenia wewnątrz CardView
        val textView = TextView(requireContext()).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
            text = timeText
            textSize = 20f
            setTextColor(resources.getColor(android.R.color.white, requireContext().theme)) // Kolor tekstu
            textAlignment = View.TEXT_ALIGNMENT_CENTER // Wyśrodkowanie tekstu
            setPadding(16, 16, 16, 16) // Marginesy wewnętrzne
        }

        // Dodajemy TextView do CardView
        cardView.addView(textView)
        return cardView
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}