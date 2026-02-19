@extends('layouts.adminSB')
@section('title', 'Quickbite - Create Transaction')

@section('content')
<div class="container">
    <h2>Tambah Transaksi</h2>

    @if ($errors->any())
        <div class="alert alert-danger">
            <ul>
                @foreach ($errors->all() as $error)
                    <li>{{ $error }}</li>
                @endforeach
            </ul>
        </div>
    @endif

    <form action="{{ route('admin.transactions.store') }}" method="POST">
        @csrf

        <div class="mb-3">
            <label for="subtotal" class="form-label">Subtotal</label>
            <input type="number" name="subtotal" class="form-control" required>
        </div>

        <div class="mb-3">
            <label for="discount" class="form-label">Discount</label>
            <input type="number" name="discount" class="form-control">
        </div>

        <div class="mb-3">
            <label for="total" class="form-label">Total</label>
            <input type="number" name="total" class="form-control" required>
        </div>

        <div class="mb-3">
            <label for="order_type" class="form-label">Tipe Pesanan</label>
            <select name="order_type" class="form-control">
                <option value="take_away">Take Away</option>
                <option value="dine_in">Dine In</option>
            </select>
        </div>

        <div class="mb-3">
            <label for="payment_type" class="form-label">Tipe Pembayaran</label>
            <select name="payment_type" class="form-control">
                <option value="QRIS">QRIS</option>
                <option value="e_wallet">E-Wallet</option>
            </select>
        </div>

        <div class="mb-3">
            <label for="status" class="form-label">Status</label>
            <select name="status" class="form-control">
                <option value="pending">Pending</option>
                <option value="processing">Processing</option>
                <option value="ready">Ready</option>
                <option value="cancelled">Cancelled</option>
            </select>
        </div>

        <div class="mb-3">
            <label for="users_id" class="form-label">User ID</label>
            <input type="number" name="users_id" class="form-control" required>
        </div>

        <button type="submit" class="btn btn-primary">Simpan</button>
        <a href="{{ route('admin.transactions.index') }}" class="btn btn-secondary">Kembali</a>
    </form>
</div>
@endsection
