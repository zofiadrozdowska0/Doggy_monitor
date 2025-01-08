package com.example.dogmonitor

import android.content.Context
import android.content.SharedPreferences

object SettingsPreferences {
    private const val PREFS_NAME = "SettingsPrefs"
    private const val NOTIFICATIONS_ENABLED_KEY = "notifications_enabled"
    private const val NOTIFICATIONS_INDICATORS_HAPPY_KEY = "notifications_indicator_happy"
    private const val NOTIFICATIONS_INDICATORS_SAD_KEY = "notifications_indicator_sad"
    private const val NOTIFICATIONS_INDICATORS_ANGRY_KEY = "notifications_indicator_angry"
    private const val NOTIFICATIONS_INDICATORS_HUNGRY_KEY = "notifications_indicator_hungry"
    private const val SERVER_ADDRESS_KEY = "server_address"
    private const val VIDEOCLIP_LENGHT_KEY = "videoclip_lenght"

    private const val VIDEOCLIP_INDICATOR_HAPPY_KEY = "videoclip_indicator_happy"
    private const val VIDEOCLIP_INDICATOR_SAD_KEY = "videoclip_indicator_sad"
    private const val VIDEOCLIP_INDICATOR_ANGRY_KEY = "videoclip_indicator_angry"
    private const val VIDEOCLIP_INDICATOR_HUNGRY_KEY = "videoclip_indicator_hungry"

    private const val DOG_BREED_KEY = "dog_breed"



    private lateinit var prefs: SharedPreferences

    fun init(context: Context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    var notificationsEnabled: Boolean
        get() = prefs.getBoolean(NOTIFICATIONS_ENABLED_KEY, false)
        set(value) = prefs.edit().putBoolean(NOTIFICATIONS_ENABLED_KEY, value).apply()

    var notificationsIndicatorHappy: Boolean
        get() = prefs.getBoolean(NOTIFICATIONS_INDICATORS_HAPPY_KEY, false)
        set(value) = prefs.edit().putBoolean(NOTIFICATIONS_INDICATORS_HAPPY_KEY, value).apply()

    var notificationsIndicatorSad: Boolean
        get() = prefs.getBoolean(NOTIFICATIONS_INDICATORS_SAD_KEY, false)
        set(value) = prefs.edit().putBoolean(NOTIFICATIONS_INDICATORS_SAD_KEY, value).apply()

    var notificationsIndicatorAngry: Boolean
        get() = prefs.getBoolean(NOTIFICATIONS_INDICATORS_ANGRY_KEY, false)
        set(value) = prefs.edit().putBoolean(NOTIFICATIONS_INDICATORS_ANGRY_KEY, value).apply()

    var notificationsIndicatorHungry: Boolean
        get() = prefs.getBoolean(NOTIFICATIONS_INDICATORS_HUNGRY_KEY, false)
        set(value) = prefs.edit().putBoolean(NOTIFICATIONS_INDICATORS_HUNGRY_KEY, value).apply()

    var videoclipLenght: Int
        get() = prefs.getInt(VIDEOCLIP_LENGHT_KEY, 0)
        set(value) = prefs.edit().putInt(VIDEOCLIP_LENGHT_KEY, value).apply()

    var videoclipIndicatorHappy: Boolean
        get() = prefs.getBoolean(VIDEOCLIP_INDICATOR_HAPPY_KEY, false)
        set(value) = prefs.edit().putBoolean(VIDEOCLIP_INDICATOR_HAPPY_KEY, value).apply()

    var videoclipIndicatorAngry: Boolean
        get() = prefs.getBoolean(VIDEOCLIP_INDICATOR_ANGRY_KEY, false)
        set(value) = prefs.edit().putBoolean(VIDEOCLIP_INDICATOR_ANGRY_KEY, value).apply()

    var videoclipIndicatorSad: Boolean
        get() = prefs.getBoolean(VIDEOCLIP_INDICATOR_SAD_KEY, false)
        set(value) = prefs.edit().putBoolean(VIDEOCLIP_INDICATOR_SAD_KEY, value).apply()

    var videoclipIndicatorHungry: Boolean
        get() = prefs.getBoolean(VIDEOCLIP_INDICATOR_HUNGRY_KEY, false)
        set(value) = prefs.edit().putBoolean(VIDEOCLIP_INDICATOR_HUNGRY_KEY, value).apply()

    var server_address: String
        get() = prefs.getString("server_address", "") ?: ""
        set(value) {
            prefs.edit().putString("server_address", value).apply()
        }

    var dog_breed: Int
        get() = prefs.getInt(DOG_BREED_KEY, 0)
        set(value) = prefs.edit().putInt(DOG_BREED_KEY,value).apply()


}