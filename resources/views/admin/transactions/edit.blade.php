@extends('layouts.adminSB')
@section('title', 'Quickbite - Edit Transaction')

@section('content')
<div class="container">
    <h2>Edit Transaksi</h2>

    @if ($errors->any())
        <div class="alert alert-danger">
            <ul>
                @foreach ($errors->all() as $error)
                    <li>{{ $error }}</li>
                @endforeach
            </ul>
        </div>
    @endif

    <form action="{{ route('admin.transactions.update', $transaction->id) }}" method="POST">
        @csrf
        @method('PUT')

        <div class="mb-3">
            <label for="subtotal" class="form-label">Subtotal</label>
            <input type="number" name="subtotal" class="form-control" value="{{ old('subtotal', $transaction->subtotal) }}" required>
        </div>

        <div class="mb-3">
            <label for="discount" class="form-label">Discount</label>
            <input type="number" name="discount" class="form-control" value="{{ old('discount', $transaction->discount) }}">
        </div>

        <div class="mb-3">
            <label for="total" class="form-label">Total</label>
            <input type="number" name="total" class="form-control" value="{{ old('total', $transaction->total) }}" required>
        </div>

        <div class="mb-3">
            <label for="order_type" class="form-label">Tipe Pesanan</label>
            <select name="order_type" class="form-control">
                <option value="take_away" {{ $transaction->order_type == 'take_away' ? 'selected' : '' }}>Take Away</option>
                <option value="dine_in" {{ $transaction->order_type == 'dine_in' ? 'selected' : '' }}>Dine In</option>
            </select>
        </div>

        <div class="mb-3">
            <label for="payment_type" class="form-label">Tipe Pembayaran</label>
            <select name="payment_type" class="form-control">
                <option value="QRIS" {{ $transaction->payment_type == 'QRIS' ? 'selected' : '' }}>QRIS</option>
                <option value="e_wallet" {{ $transaction->payment_type == 'e_wallet' ? 'selected' : '' }}>E-Wallet</option>
            </select>
        </div>

        <div class="mb-3">
            <label for="status" class="form-label">Status</label>
            <select name="status" class="form-control">
                <option value="pending" {{ $transaction->status == 'pending' ? 'selected' : '' }}>Pending</option>
                <option value="processing" {{ $transaction->status == 'processing' ? 'selected' : '' }}>Processing</option>
                <option value="ready" {{ $transaction->status == 'ready' ? 'selected' : '' }}>Ready</option>
                <option value="cancelled" {{ $transaction->status == 'cancelled' ? 'selected' : '' }}>Cancelled</option>
            </select>
        </div>

        <div class="mb-3">
            <label for="users_id" class="form-label">User ID</label>
            <input type="number" name="users_id" class="form-control" value="{{ old('users_id', $transaction->users_id) }}" required>
        </div>

        <button type="submit" class="btn btn-primary">Simpan Perubahan</button>
        <a href="{{ route('admin.transactions.index') }}" class="btn btn-secondary">Kembali</a>
    </form>
</div>
@endsection
