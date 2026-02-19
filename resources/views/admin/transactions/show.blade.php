@extends('layouts.adminSB')
@section('title', 'Quickbite - Transaction Details')

@section('content')
<div class="container">
    <h2>Detail Transaksi #{{ $transaction->id }}</h2>

    <div class="mb-3">
        <strong>Nama Customer:</strong> {{ $transaction->user->name ?? '-' }}<br>
        <strong>Tanggal:</strong> {{ $transaction->created_at->format('d M Y H:i') }}<br>
        <strong>Metode:</strong> {{ $transaction->order_type }}<br>
        <strong>Pembayaran:</strong> {{ $transaction->payment_type }}
    </div>

    <h4>Daftar Pesanan</h4>
    <table class="table table-bordered">
        <thead>
            <tr>
                <th>Menu</th>
                <th>Jumlah</th>
                <th>Harga Satuan</th>
                <th>Total Harga</th>
            </tr>
        </thead>
        <tbody>
            @php $grandTotal = 0; @endphp
            @foreach ($transaction->orders as $order)
                @php
                    $subtotal = $order->quantity * $order->menu->price;
                    $grandTotal += $subtotal;
                @endphp
                <tr>
                    <td>{{ $order->menu->name }}</td>
                    <td>{{ $order->quantity }}</td>
                    <td>Rp{{ number_format($order->menu->price, 0, ',', '.') }}</td>
                    <td>Rp{{ number_format($subtotal, 0, ',', '.') }}</td>
                </tr>
            @endforeach
            <tr>
                <td colspan="3"><strong>Total</strong></td>
                <td><strong>Rp{{ number_format($grandTotal, 0, ',', '.') }}</strong></td>
            </tr>
        </tbody>
    </table>

    <a href="{{ route('admin.transactions.index') }}" class="btn btn-secondary">Kembali</a>
</div>
@endsection
