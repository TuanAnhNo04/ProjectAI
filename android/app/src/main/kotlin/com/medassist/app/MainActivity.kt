package com.medassist.app

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.CancellationSignal
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.util.concurrent.atomic.AtomicBoolean

/**
 * FLAG_SECURE: chặn chụp/quay màn hình và ẩn nội dung màn hình (tên thuốc, hồ sơ
 * sức khoẻ) khỏi ảnh xem trước trong danh sách app gần đây — dữ liệu y tế không nên
 * lộ qua ảnh chụp màn hình hay bản ghi màn hình của hệ điều hành/launcher khác.
 *
 * MethodChannel "medassist/device": lấy vị trí (chỉ trả về cho Flutter, không gửi
 * đi đâu, không ghi log), mở trình quay số (ACTION_DIAL — không cần CALL_PHONE),
 * và mở ứng dụng bản đồ bằng intent geo:.
 */
class MainActivity : FlutterActivity() {
    private val channelName = "medassist/device"
    private val locationRequestCode = 4711
    private val locationTimeoutMs = 10_000L
    private var pendingLocationResult: MethodChannel.Result? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        window.setFlags(
            WindowManager.LayoutParams.FLAG_SECURE,
            WindowManager.LayoutParams.FLAG_SECURE,
        )
        super.onCreate(savedInstanceState)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result -> handle(call, result) }
    }

    private fun handle(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "getLocation" -> getLocation(result)
            "dial" -> {
                val raw = call.argument<String>("number") ?: ""
                val number = raw.filter { it.isDigit() || it == '+' }
                if (number.isEmpty()) {
                    result.error("invalid_argument", "empty number", null)
                } else {
                    startSafely(Intent(Intent.ACTION_DIAL, Uri.parse("tel:$number")), result)
                }
            }
            "openMap" -> {
                val lat = call.argument<Double>("lat")
                val lng = call.argument<Double>("lng")
                val label = call.argument<String>("label") ?: ""
                if (lat == null || lng == null) {
                    result.error("invalid_argument", "missing coordinates", null)
                } else {
                    val uri = Uri.parse("geo:$lat,$lng?q=$lat,$lng(${Uri.encode(label)})")
                    startSafely(Intent(Intent.ACTION_VIEW, uri), result)
                }
            }
            "openUrl" -> {
                // Chỉ https: chặn file:, intent:, javascript:, http không mã hoá.
                val uri = Uri.parse(call.argument<String>("url") ?: "")
                if (uri.scheme?.lowercase() != "https" || uri.host.isNullOrEmpty()) {
                    result.error("invalid_argument", "only https urls", null)
                } else {
                    startSafely(Intent(Intent.ACTION_VIEW, uri), result)
                }
            }
            "searchMap" -> {
                val query = call.argument<String>("query") ?: ""
                val uri = Uri.parse("geo:0,0?q=${Uri.encode(query)}")
                startSafely(Intent(Intent.ACTION_VIEW, uri), result)
            }
            else -> result.notImplemented()
        }
    }

    private fun startSafely(intent: Intent, result: MethodChannel.Result) {
        try {
            startActivity(intent)
            result.success(null)
        } catch (e: ActivityNotFoundException) {
            result.error("no_activity", "no app can handle this intent", null)
        }
    }

    // --- Vị trí -------------------------------------------------------------

    private fun hasPermission(permission: String) =
        checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

    private fun hasAnyLocationPermission() =
        hasPermission(Manifest.permission.ACCESS_FINE_LOCATION) ||
            hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION)

    private fun getLocation(result: MethodChannel.Result) {
        if (hasAnyLocationPermission()) {
            fetchLocation(result)
            return
        }
        if (pendingLocationResult != null) {
            result.error("unavailable", "location request already in progress", null)
            return
        }
        pendingLocationResult = result
        requestPermissions(
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION,
            ),
            locationRequestCode,
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != locationRequestCode) return
        val result = pendingLocationResult ?: return
        pendingLocationResult = null
        if (grantResults.any { it == PackageManager.PERMISSION_GRANTED }) {
            fetchLocation(result)
        } else {
            result.error("permission_denied", "location permission denied", null)
        }
    }

    @Suppress("DEPRECATION", "MissingPermission")
    private fun fetchLocation(result: MethodChannel.Result) {
        val manager = getSystemService(Context.LOCATION_SERVICE) as? LocationManager
        if (manager == null) {
            result.error("unavailable", "no location service", null)
            return
        }
        val gpsOn = isProviderOn(manager, LocationManager.GPS_PROVIDER)
        val networkOn = isProviderOn(manager, LocationManager.NETWORK_PROVIDER)
        val locationOn = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            manager.isLocationEnabled
        } else {
            gpsOn || networkOn
        }
        if (!locationOn || (!gpsOn && !networkOn)) {
            result.error("location_off", "location is turned off", null)
            return
        }
        val fine = hasPermission(Manifest.permission.ACCESS_FINE_LOCATION)
        val provider = when {
            fine && gpsOn -> LocationManager.GPS_PROVIDER
            networkOn -> LocationManager.NETWORK_PROVIDER
            else -> LocationManager.GPS_PROVIDER
        }

        val replied = AtomicBoolean(false)
        fun reply(location: Location?) {
            if (!replied.compareAndSet(false, true)) return
            val loc = location ?: lastKnown(manager)
            if (loc == null) {
                result.error("unavailable", "location unavailable", null)
            } else {
                result.success(mapOf("lat" to loc.latitude, "lng" to loc.longitude))
            }
        }

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val cancel = CancellationSignal()
                mainHandler.postDelayed({
                    cancel.cancel()
                    reply(null)
                }, locationTimeoutMs)
                manager.getCurrentLocation(provider, cancel, mainExecutor) { reply(it) }
            } else {
                val listener = object : LocationListener {
                    override fun onLocationChanged(location: Location) {
                        reply(location)
                    }

                    override fun onStatusChanged(p: String?, s: Int, e: Bundle?) {}
                    override fun onProviderEnabled(p: String) {}
                    override fun onProviderDisabled(p: String) {}
                }
                mainHandler.postDelayed({
                    manager.removeUpdates(listener)
                    reply(null)
                }, locationTimeoutMs)
                manager.requestSingleUpdate(provider, listener, Looper.getMainLooper())
            }
        } catch (e: SecurityException) {
            if (replied.compareAndSet(false, true)) {
                result.error("permission_denied", "location permission denied", null)
            }
        } catch (e: IllegalArgumentException) {
            reply(null)
        }
    }

    private fun isProviderOn(manager: LocationManager, provider: String) = try {
        manager.isProviderEnabled(provider)
    } catch (e: Exception) {
        false
    }

    @Suppress("MissingPermission")
    private fun lastKnown(manager: LocationManager): Location? {
        val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
        return providers
            .mapNotNull {
                try {
                    manager.getLastKnownLocation(it)
                } catch (e: Exception) {
                    null
                }
            }
            .maxByOrNull { it.time }
    }
}
